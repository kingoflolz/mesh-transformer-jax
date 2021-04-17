import transformers
from lm_eval.base import LM
from tqdm import tqdm
import numpy as np

from tasks.util import sample_batch


class EvalHarnessAdaptor(LM):
    def greedy_until(self, requests):
        raise Exception("unimplemented")

    def __init__(self, tpu_cluster, seq, batch):
        self.tpu = tpu_cluster
        self.seq = seq
        self.batch = batch

        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
        self.tokenizer.pad_token = "<|endoftext|>"

        assert self.tokenizer.encode('hello\n\nhello') == [31373, 198, 198, 31373]

    def request_iter(self, requests):
        for ctx, cont in requests:
            ctx_tokens = self.tokenizer.encode("<|endoftext|>" + ctx)
            cont_tokens = self.tokenizer.encode(cont)

            all_tokens = ctx_tokens + cont_tokens
            all_tokens = np.array(all_tokens)[-self.seq:]  # truncate sequence at seq length

            provided_ctx = len(all_tokens) - 1
            pad_amount = self.seq - provided_ctx

            yield {
                "obs": np.pad(all_tokens[:-1], ((pad_amount, 0),), constant_values=50256),
                "target": np.pad(all_tokens[1:], ((pad_amount, 0),), constant_values=50256),
                "ctx_length": provided_ctx,
                "eval_mask": np.arange(0, self.seq) > (self.seq - len(cont_tokens) - 1),
            }

    def loglikelihood(self, requests):
        output = []

        r = list(self.request_iter(requests))

        for b in tqdm(sample_batch(r, self.batch), desc="LM eval harness", total=len(r) // self.batch):
            out = self.tpu.eval(b)

            for loss, correct in zip(out["mask_loss"], out["each_correct"]):
                output.append((float(-loss), bool(correct)))

        return output
