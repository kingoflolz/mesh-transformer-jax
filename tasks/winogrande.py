import json
import numpy as np
import requests
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from tasks.util import sample_batch


class WinograndeTask:
    def __init__(self, max_ctx):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        samples = requests.get("https://gist.githubusercontent.com/kingoflolz/e79015778958bcb1775b8dc383d1efa0/raw/1fd9d4940b97efa9ad8948cbe7ddb6a175f8487e/dev.jsonl").text

        samples = [json.loads(line) for line in samples.splitlines()]

        self.correct_examples = []
        self.wrong_examples = []

        for i in samples:
            answer = int(i["answer"])
            sentence = i["sentence"]

            if answer == 1:
                correct_sentence = sentence.replace("_", i["option1"])
                wrong_sentence = sentence.replace("_", i["option2"])
            elif answer == 2:
                correct_sentence = sentence.replace("_", i["option2"])
                wrong_sentence = sentence.replace("_", i["option1"])
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
        winogrande_total = 0
        winogrande_correct = 0

        for correct, wrong in tqdm(zip(
                self.sample_correct(bs),
                self.sample_wrong(bs),
        ), total=len(self.correct_examples) // bs, desc="winogrande eval"):
            correct_out = tpu.eval(correct)
            wrong_out = tpu.eval(wrong)

            winogrande_total += correct_out["total"]
            winogrande_correct += np.sum(correct_out["mask_loss"] < wrong_out["mask_loss"])

        return {
            "winogrande/acc": winogrande_correct / winogrande_total
        }


if __name__ == "__main__":
    l = WinograndeTask(2048)

    for c, w in zip(l.sample_correct(16), l.sample_wrong(16)):
        print(c, w)
