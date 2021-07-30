import argparse
import os
import re
import random

from pathlib import Path

import ftfy
import tensorflow as tf
from lm_dataformat import Reader
from transformers import GPT2TokenizerFast
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="""
    Converts a text dataset into the training data format expected by the model.

    Adapted from the script create_tfrecords.py in the gpt-neo repo.

    - Your text dataset:
        - can be provided as .txt files, or as an archive (.tar.gz, .xz, jsonl.zst).
        - can be one file or multiple
            - using a single large file may use too much memory and crash - if this occurs, split the file up into a few files
        - the model's end-of-text separator is added between the contents of each file
        - if the string '<|endoftext|>' appears inside a file, it is treated as the model's end-of-text separator (not the actual string '<|endoftext|>')
            - this behavior can be disabled with --treat-eot-as-text

    This script creates a single .tfrecords file as output
        - Why: the model's data loader ignores "trailing" data (< 1 batch) at the end of a .tfrecords file
            - this causes data loss if you have many .tfrecords files
        - This is probably not appropriate for very large datasets
    """,
    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("input_dir", type=str, help="Path to where your files are located.")
    parser.add_argument("name", type=str,
                        help="Name of output file will be {name}_{seqnum}.tfrecords, where seqnum is total sequence count")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory (default: directory)")

    cleaning_args = parser.add_argument_group('data cleaning arguments')

    cleaning_args.add_argument("--normalize-with-ftfy", action="store_true", help="Normalize text with ftfy")
    cleaning_args.add_argument("--normalize-with-wikitext-detokenize",
                               action="store_true", help="Use wikitext detokenizer")
    minu_help = "Exclude repetitive documents made up of < MIN_UNIQUE_TOKENS unique tokens. These can produce large gradients."
    minu_help += " Set <= 0 to disable. If enabled, 200 is a good default value. (Default: 0)"
    cleaning_args.add_argument("--min-unique-tokens", type=int, default=0,
                               help=minu_help)
    eot_text_help = "Treats the string '<|endoftext|>' inside files as text rather than a document separator."
    eot_text_help += " Not appropriate if your dataset is already concatenate on '<|endoftext|>'."
    cleaning_args.add_argument("--treat-eot-as-text",
                               default=False, action="store_true",
                               help=eot_text_help)

    shuffle_pack_args = parser.add_argument_group('data shuffling/packing arguments')
    repack_ep_help = "Repeat the data N_REPACK_EPOCHS times, shuffled differently in each repetition. Recommended for multi-epoch training (set this to your intended number of epochs)."
    shuffle_pack_args.add_argument("--n-repack-epochs",
                                   type=int, default=1,
                                   help=repack_ep_help
                                   )
    shuffle_pack_args.add_argument("--seed", type=int, default=10,
                                   help="random seed for shuffling data (default: 10)")
    shuffle_pack_args.add_argument("--preserve-data-order",
                                   default=False, action="store_true",
                                   help="Disables shuffling, so the input and output data have the same order.")

    misc_args = parser.add_argument_group('miscellaneous arguments')
    misc_args.add_argument("--verbose",
                           default=False, action="store_true",
                           help="Prints extra information, such as the text removed by --min-unique-tokens")

    args = parser.parse_args()

    if not args.input_dir.endswith("/"):
        args.input_dir = args.input_dir + "/"

    return args


def wikitext_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string


def _int64_feature(value):
    """
    Returns an int64_list from a bool / enum / int / uint.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_to_file(writer, data):
    """
    writes data to tfrecord file
    """
    feature = {
        "text": _int64_feature(data)
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(tf_example.SerializeToString())


def write_tfrecord(files, fp):
    chunks = files
    files_per = len(files)

    with tf.io.TFRecordWriter(fp) as writer:
        for f in files:
            write_to_file(writer, f)


def split_list(l, n):
    # splits list/string into n size chunks
    return [l[i:i + n] for i in range(0, len(l), n)]


def enforce_min_unique(seqs, min_unique_tokens, enc, verbose=False):
    for seq in tqdm(seqs, mininterval=1, smoothing=0):
        if len(set(seq)) >= min_unique_tokens:
            yield seq
        elif verbose:
            text = enc.decode(seq)
            print(f"excluding with {len(set(seq))} unique tokens:\n\n{repr(text)}\n\n")


def split_on_interior_eot(doc, encoder, disable=False):
    if disable:
        return [doc]
    return [d for d in doc.split(encoder.eos_token) if d]


def archive_to_tokens(f, encoder, args, prefix=[]):
    # Generator that yields the contents of the files in an archive
    # if data_to_prepend is not None, prepend data_to_prepend + a EOS separator to the encoded data
    reader = Reader(f)
    for file_doc in reader.stream_data(threaded=False):
        for doc in split_on_interior_eot(file_doc, encoder, disable=args.treat_eot_as_text):
            if args.normalize_with_ftfy:  # fix text with ftfy if specified
                doc = ftfy.fix_text(doc, normalization='NFKC')
            if args.normalize_with_wikitext_detokenize:
                doc = wikitext_detokenizer(doc)
            doc = encoder.encode(doc) + [encoder.eos_token_id]  # read document from lmd and append separator token
            yield split_list(prefix + doc, 2049)  # split into n_ctx + 1 size chunks
            prefix = []


def get_files(input_dir):
    filetypes = ["jsonl.zst", ".txt", ".xz", ".tar.gz"]
    files = [list(Path(input_dir).glob(f"*{ft}")) for ft in filetypes]
    # flatten list of list -> list and stringify Paths
    return [str(item) for sublist in files for item in sublist]


def create_tfrecords(files, args):
    GPT2TokenizerFast.max_model_input_sizes['gpt2'] = 1e20  # disables a misleading warning
    enc = GPT2TokenizerFast.from_pretrained('gpt2')

    random.seed(args.seed)

    data_to_prepend = []
    all_sequences_across_epochs = []

    ep_len = None

    for ep_ix in range(args.n_repack_epochs):
        tokenized_files_array = []

        if args.preserve_data_order:
            files = sorted(files)
        else:
            random.shuffle(files)

        print(f'starting epoch {ep_ix}\n\t{len(all_sequences_across_epochs)} sequences so far\n\t{len(data_to_prepend)} tokens rolled over from last epoch\n\tfirst file this ep is {files[0]}')

        for f in tqdm(files, mininterval=10, smoothing=0):
            for tokenized_files in archive_to_tokens(f, enc, args, prefix=data_to_prepend):
                # if the last chunk < chunk size, take it and append it to the beginning of the next file
                data_to_prepend = []
                n_tokens = len(tokenized_files[-1])
                if n_tokens < 2049:
                    data = tokenized_files.pop(-1)
                    data_to_prepend = data

                tokenized_files_array.extend(tokenized_files)

        if not args.preserve_data_order:
            random.shuffle(tokenized_files_array)

        if args.min_unique_tokens > 0:
            tokenized_files_array = list(enforce_min_unique(tokenized_files_array, args.min_unique_tokens, enc, args.verbose))

        all_sequences_across_epochs.extend(tokenized_files_array)

        if ep_ix == 0:
            ep_len = len(tokenized_files_array)

    total_sequence_len = len(all_sequences_across_epochs)

    fp = os.path.join(args.output_dir, f"{args.name}_{total_sequence_len}.tfrecords")
    write_tfrecord(all_sequences_across_epochs, fp)


if __name__ == "__main__":
    args = parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    files = get_files(args.input_dir)

    results = create_tfrecords(files, args)
