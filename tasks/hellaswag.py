import json
import random
import re

import numpy as np
import requests
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from tasks.util import sample_batch


class HellaSwagTask:
    def __init__(self, max_ctx, first = None):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        samples = requests.get(
            "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl").text

        samples = [json.loads(line) for line in samples.splitlines()]

        random.seed(42)
        random.shuffle(samples)

        if first:
            samples = samples[:first]

        self.neg_examples = [[], [], []]
        self.pos_examples = []

        for sample in samples:
            endings = []
            for ending in sample["endings"]:
                text = sample["ctx"] + " " + ending

                text = text.strip()
                # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
                text = text.replace(" [title]", ". ")
                text = re.sub('\\[.*?\\]', '', text)
                text = text.replace("  ", " ")

                sentence = self.tokenizer(text)["input_ids"]
                pad_amount = max_ctx - len(sentence) + 1

                endings.append({
                    "obs": np.pad(sentence[:-1], ((pad_amount, 0),)),
                    "target": np.pad(sentence[1:], ((pad_amount, 0),)),
                    "ctx_length": len(sentence) - 1,
                    "eval_mask": np.arange(0, max_ctx) > pad_amount - 1,
                })

            self.pos_examples.append(endings.pop(sample["label"]))

            for l, s in zip(self.neg_examples, endings):
                l.append(s)

    @staticmethod
    def clean():
        return 1

    def run(self, bs, tpu):
        hella_total = 0
        hella_correct = 0

        iterators = [sample_batch(i, bs) for i in [self.pos_examples] + self.neg_examples]

        for pos, *neg in tqdm(zip(*iterators), total=len(self.pos_examples) // bs, desc="hellaswag eval"):
            neg_losses = []
            for choice in neg:
                out = tpu.eval(choice)
                neg_losses.append(out["mask_loss"])

            best_neg = np.min(np.array(neg_losses), axis=0)

            correct_out = tpu.eval(pos)
            hella_total += correct_out["total"]
            hella_correct += np.sum(correct_out["mask_loss"] < best_neg)

        return {
            "hellaswag/acc": hella_correct / hella_total
        }


if __name__ == "__main__":
    l = HellaSwagTask(2048)

    for c, w in zip(l.sample_correct(16), l.sample_wrong(16)):
        print(c, w)
