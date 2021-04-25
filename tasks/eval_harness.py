from functools import partial

import transformers
from lm_eval.base import LM
from tqdm import tqdm
import numpy as np

from tasks.util import sample_batch
import multiprocessing

tokenizer = None


def process_init():
    global tokenizer
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = "<|endoftext|>"

    assert tokenizer.encode('hello\n\nhello') == [31373, 198, 198, 31373]


def process_request(x, seq):
    global tokenizer

    ctx, cont = x

    ctx_tokens = tokenizer.encode("<|endoftext|>" + ctx)
    cont_tokens = tokenizer.encode(cont)

    all_tokens = ctx_tokens + cont_tokens
    all_tokens = np.array(all_tokens)[-seq:]  # truncate sequence at seq length

    provided_ctx = len(all_tokens) - 1
    pad_amount = seq - provided_ctx

    return {
        "obs": np.pad(all_tokens[:-1], ((pad_amount, 0),), constant_values=50256),
        "target": np.pad(all_tokens[1:], ((pad_amount, 0),), constant_values=50256),
        "ctx_length": provided_ctx,
        "eval_mask": np.arange(0, seq) > (seq - len(cont_tokens) - 1),
    }


class EvalHarnessAdaptor(LM):
    def greedy_until(self, requests):
        raise Exception("unimplemented")

    def __init__(self, tpu_cluster, seq, batch):
        self.tpu = tpu_cluster
        self.seq = seq
        self.batch = batch

        self.pool = multiprocessing.Pool(initializer=process_init)

    def convert_requests(self, requests):
        return list(tqdm(self.pool.imap(partial(process_request, seq=self.seq), requests), desc="request conversion"))

    def loglikelihood(self, requests):
        output = []

        r = self.convert_requests(requests)

        for b in tqdm(sample_batch(r, self.batch), desc="LM eval harness", total=len(r) // self.batch):
            out = self.tpu.eval(b)

            for loss, correct in zip(out["mask_loss"], out["each_correct"]):
                output.append((float(-loss), bool(correct)))

        return output
