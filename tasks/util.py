from itertools import zip_longest

import numpy as np


def grouper(n, iterable, fillvalue):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def sample_batch(examples, bs, zero_example_shape):
    zero_example = {
        "obs": np.zeros_like(zero_example_shape["obs"]),
        "target": np.zeros_like(zero_example_shape["target"]),
        "eval_mask": np.zeros_like(zero_example_shape["eval_mask"]),
        "ctx_length": 0,
    }

    for batch in grouper(bs, examples, zero_example):
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