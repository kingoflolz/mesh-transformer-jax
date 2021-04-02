import json
import numpy as np
import requests
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from tasks.util import sample_batch


class PIQATask:
    def __init__(self, max_ctx):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        samples = requests.get("https://yonatanbisk.com/piqa/data/valid.jsonl").text
        labels = requests.get("https://yonatanbisk.com/piqa/data/valid-labels.lst").text

        samples = [json.loads(line) for line in samples.splitlines()]
        labels = labels.splitlines()

        self.correct_examples = []
        self.wrong_examples = []

        for label, sample in zip(labels, samples):
            if label == "0":
                correct_sentence = sample["goal"] + " " + sample["sol1"]
                wrong_sentence = sample["goal"] + " " + sample["sol2"]
            elif label == '1':
                correct_sentence = sample["goal"] + " " + sample["sol2"]
                wrong_sentence = sample["goal"] + " " + sample["sol1"]
            else:
                raise Exception("uh oh")

            correct_sentence = self.tokenizer(correct_sentence)["input_ids"]
            wrong_sentence = self.tokenizer(wrong_sentence)["input_ids"]

            pad_amount = max_ctx - len(correct_sentence) + 1

            self.correct_examples.append({
                "obs": np.pad(correct_sentence[:-1], ((pad_amount, 0),)),
                "target": np.pad(correct_sentence[1:], ((pad_amount, 0),)),
                "ctx_length": len(correct_sentence) - 1,
                "eval_mask": np.arange(0, max_ctx) > pad_amount - 1,
            })

            pad_amount = max_ctx - len(wrong_sentence) + 1

            self.wrong_examples.append({
                "obs": np.pad(wrong_sentence[:-1], ((pad_amount, 0),)),
                "target": np.pad(wrong_sentence[1:], ((pad_amount, 0),)),
                "ctx_length": len(wrong_sentence) - 1,
                "eval_mask": np.arange(0, max_ctx) > pad_amount - 1,
            })

    def sample_correct(self, bs):
        return sample_batch(self.correct_examples, bs)

    def sample_wrong(self, bs):
        return sample_batch(self.wrong_examples, bs)

    def run(self, bs, tpu):
        piqa_total = 0
        piqa_correct = 0

        for correct, wrong in tqdm(zip(
                self.sample_correct(bs),
                self.sample_wrong(bs),
        ), total=len(self.correct_examples) // bs, desc="piqa eval"):
            correct_out = tpu.eval(correct)
            wrong_out = tpu.eval(wrong)

            piqa_total += correct_out["total"]
            piqa_correct += np.sum(correct_out["mask_loss"] < wrong_out["mask_loss"])

        return {
            "piqa/acc": piqa_correct / piqa_total
        }


if __name__ == "__main__":
    l = PIQATask(2048)

    for c, w in zip(l.sample_correct(16), l.sample_wrong(16)):
        print(c, w)
