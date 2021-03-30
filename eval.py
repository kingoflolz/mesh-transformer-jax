import argparse
import json
import time

import numpy as np
import wandb
from tqdm import tqdm

from lambada import LambadaTask
from mesh_transformer.build_model import build_model
from tfrecord_loader import TFRecordNewInputs


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpu", type=str, help="Name of TPU to train on.")
    parser.add_argument("--tpu_region", type=str, help="Region of TPU to train on.")
    parser.add_argument("--preemptible", action="store_true")

    parser.add_argument("--config", type=str, default=None, help="Config file location")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    params = json.load(open(args.config))

    tpu_name = args.tpu
    region = args.tpu_region
    preemptible = args.preemptible

    gradient_accumulation_steps = params.get("gradient_accumulation_steps", 1)
    per_replica_batch = params["per_replica_batch"]
    tpu_size = params["tpu_size"]
    cores_per_replica = params["cores_per_replica"]

    bucket = params["bucket"]
    model_dir = params["model_dir"]
    layers = params["layers"]
    d_model = params["d_model"]
    n_heads = params["n_heads"]
    n_vocab = params["n_vocab"]
    seq = params["seq"]
    norm = params["norm"]

    t = build_model(params, tpu_name, region, preemptible)

    val_dataset = LambadaTask(2048)

    step, aux = t.load(bucket, model_dir)

    start = time.time()
    t.eval(next(val_dataset.sample_batch(per_replica_batch * tpu_size // cores_per_replica)))
    print(f"Eval fn compiled in {time.time() - start:.06}s")

    total = 0
    correct = 0

    for batch in tqdm(val_dataset.sample_batch(per_replica_batch * tpu_size // cores_per_replica)):
        out = t.eval(batch)
        total += out["total"]
        correct += out["correct"]

    print("total", total)
    print("correct", correct)
