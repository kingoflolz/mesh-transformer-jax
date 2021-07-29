from functools import partial

import transformers
from lm_eval.base import LM
from tqdm import tqdm
import numpy as np

from tasks.util import sample_batch, shrink_seq
import multiprocessing
import ftfy

tokenizer = None


def process_init():
    global tokenizer
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.model_max_length = int(1e30)
    tokenizer.pad_token = "<|endoftext|>"

    assert tokenizer.encode('hello\n\nhello') == [31373, 198, 198, 31373]


def process_request(x, seq):
    global tokenizer

    ctx, cont = x

    ctx_tokens = tokenizer.encode("<|endoftext|>" + ftfy.fix_text(ctx, normalization="NFKC"))
    cont_tokens = tokenizer.encode(ftfy.fix_text(cont, normalization="NFKC"))

    all_tokens = ctx_tokens + cont_tokens
    all_tokens = np.array(all_tokens)[-seq:]  # truncate sequence at seq length

    provided_ctx = len(all_tokens) - 1
    pad_amount = seq - provided_ctx

    return {
        "obs": np.pad(all_tokens[:-1], ((0, pad_amount),), constant_values=50256),
        "target": np.pad(all_tokens[1:], ((0, pad_amount),), constant_values=50256),
        "ctx_length": seq,
        "eval_mask": np.logical_and(
            np.arange(0, seq) > len(all_tokens) - len(cont_tokens) - 2,
            np.arange(0, seq) < len(all_tokens) - 1
        ),
    }


class EvalHarnessAdaptor(LM):
    def greedy_until(self, requests):
        raise Exception("unimplemented")

    def loglikelihood_rolling(self, requests):
        raise Exception("unimplemented")

    def __init__(self, tpu_cluster, seq, batch, shrink, min_seq=None):
        super().__init__()
        self.tpu = tpu_cluster
        self.seq = seq
        self.batch = batch
        self.shrink = shrink
        self.min_seq = min_seq

        self.pool = multiprocessing.Pool(initializer=process_init)
        process_init()

    def convert_requests(self, requests):
        return self.pool.imap(partial(process_request, seq=self.seq), requests)

    def loglikelihood(self, requests):
        output = []

        r = self.convert_requests(requests)
        zero_example = process_request(requests[0], self.seq)

        for b in tqdm(sample_batch(r, self.batch, zero_example),
                      desc="LM eval harness",
                      total=len(requests) // self.batch):

            if self.shrink:
                b = shrink_seq(b, min_seq=self.min_seq)

            out = self.tpu.eval(b)

            for loss, correct in zip(out["mask_loss"], out["each_correct"]):
                output.append((float(-loss), bool(correct)))

        return output
