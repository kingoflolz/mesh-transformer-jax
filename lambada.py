import json
from itertools import zip_longest

import ftfy
import numpy as np
import requests
from transformers import GPT2TokenizerFast


def preprocess(text):
    return ftfy.fix_text(text, normalization="NFKC")


def grouper(n, iterable, fillvalue):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

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
        zero_example = {
            "obs": np.zeros_like(self.examples[0]["obs"]),
            "target": np.zeros_like(self.examples[0]["target"]),
            "eval_mask": np.zeros_like(self.examples[0]["eval_mask"]),
            "ctx_length": 0,
        }

        for batch in grouper(bs, self.examples, zero_example):
            batch_flattened = {
                "obs": [],
                "target": [],
                "eval_mask": [],
                "ctx_length": [],
            }

            for sample in batch:
                batch_flattened["obs"].append(sample["obs"])
                batch_flattened["target"].append(sample["target"])
                batch_flattened["eval_mask"].append(sample["eval_mask"])
                batch_flattened["ctx_length"].append(sample["ctx_length"])

            batch_flattened["obs"] = np.array(batch_flattened["obs"])
            batch_flattened["target"] = np.array(batch_flattened["target"])
            batch_flattened["eval_mask"] = np.array(batch_flattened["eval_mask"])
            batch_flattened["ctx_length"] = np.array(batch_flattened["ctx_length"])

            yield batch_flattened

if __name__ == "__main__":
    l = LambadaTask(2048)

    for i in l.sample_batch(16):
        print(i)
