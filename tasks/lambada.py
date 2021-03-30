import json
import ftfy
import numpy as np
import requests
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from tasks.util import sample_batch


def preprocess(text):
    return ftfy.fix_text(text, normalization="NFKC")


class LambadaTask:
    def __init__(self, max_ctx):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        samples = requests.get("http://eaidata.bmk.sh/data/lambada_test.jsonl").text

        samples = [preprocess(json.loads(line)["text"]) for line in samples.splitlines()]

        self.examples = []

        for i in samples:
            all_tokens = self.tokenizer(i)["input_ids"]
            words = i.split(' ')
            last_word = words[-1]
            context = " ".join(words[:-1])
            ctx_tokens = self.tokenizer(context)["input_ids"]
            last_tokens = self.tokenizer(" " + last_word)["input_ids"]

            assert all_tokens == ctx_tokens + last_tokens

            all_tokens = np.array(all_tokens)[-max_ctx:]
            provided_ctx = len(all_tokens) - 1
            pad_amount = max_ctx - provided_ctx

            self.examples.append({
                "obs": np.pad(all_tokens[:-1], ((pad_amount, 0),)),
                "target": np.pad(all_tokens[1:], ((pad_amount, 0),)),
                "ctx_length": provided_ctx,
                "eval_mask": np.arange(0, max_ctx) > (max_ctx - len(last_tokens) - 1),
            })

    def sample_batch(self, bs):
        return sample_batch(self.examples, bs)

    def run(self, bs, tpu):
        total = 0
        correct = 0
        last_correct = 0
        total_last_loss = 0

        for batch in tqdm(self.sample_batch(bs), total=len(self.examples)//bs,  desc="lambada eval"):
            out = tpu.eval(batch)
            total += out["total"]
            correct += out["correct"]
            last_correct += out["last_correct"]
            total_last_loss += out["last_loss"]

        return {
            "lambada/acc": correct / total,
            "lambada/fake_acc": last_correct / total,
            "lambada/fake_ppl": np.exp(total_last_loss / total),
        }


if __name__ == "__main__":
    l = LambadaTask(2048)

    for i in l.sample_batch(16):
        print(i)
