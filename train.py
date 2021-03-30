import argparse
import functools
import json
import multiprocessing
import time

import numpy as np
import optax
import ray
import wandb
from tqdm import tqdm

from mesh_transformer.build_model import build_model
from tfrecord_loader import TFRecordNewInputs


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpu", type=str, help="Name of TPU to train on.")
    parser.add_argument("--tpu_region", type=str, help="Region of TPU to train on.")
    parser.add_argument("--preemptible", action="store_true")

    parser.add_argument("--config", type=str, default=None, help="Config file location")

    parser.add_argument("--new", action="store_true", help="If set, deletes previous checkpoint, if it exists, and "
                                                           "starts a new training run")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    params = json.load(open(args.config))

    if args.new:
        print(f"Starting experiment {params['name']} from scratch! "
              f"all data in gs://{params['bucket']}/{params['model_dir']}/ will be deleted")
        input("Hit enter to continue")

    tpu_name = args.tpu
    region = args.tpu_region
    preemptible = args.preemptible
    clean_start = args.new

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

    val_batches = params["val_batches"]
    val_every = params["val_every"]
    ckpt_every = params["ckpt_every"]

    t = build_model(params, tpu_name, region, preemptible)

    try:
        t.save(0, bucket, model_dir, init=True, overwrite=clean_start)
        step = 0
        train_load_restore = None
    except Exception as e:
        print(f"Save failed with error {e}, trying to load instead...", e)
        step, aux = t.load(bucket, model_dir)
        train_load_restore = aux.get("train_loader", None)

        if train_load_restore is None:
            print("Failed to restore train loader state")

    train_dataset = TFRecordNewInputs(f"data/{params['train_set']}",
                                      batch_size=(
                                      gradient_accumulation_steps, per_replica_batch * tpu_size // cores_per_replica),
                                      sample_size=params['seq'],
                                      restore_state=train_load_restore)

    val_dataset = TFRecordNewInputs(f"data/{params['val_set']}",
                                    batch_size=(per_replica_batch * tpu_size // cores_per_replica,),
                                    sample_size=params['seq'])

    start = time.time()
    t.train(train_dataset.get_samples())
    print(f"Train fn compiled in {time.time() - start:.06}s")

    start = time.time()
    t.eval(val_dataset.get_samples())
    print(f"Eval fn compiled in {time.time() - start:.06}s")

    # writer = SummaryWriter(flush_secs=5)
    wandb.init(project='mesh-transformer-jax', entity="eleutherai", name=params["name"], config=params)

    while True:
        loss, last_loss = t.train(train_dataset.get_samples())
        wandb.log({'train/loss': loss, 'train/last_loss': last_loss}, step)

        if step % ckpt_every == 0 and step:
            t.save(step, bucket, model_dir, aux={"train_loader": train_dataset.get_state()}, init=False)

        if step % val_every == 0:
            val_loss = []
            for i, _ in tqdm(zip(val_dataset.sample_once(), range(val_batches)),
                             desc=f"validation for step {step}",
                             total=val_batches):
                val_loss.append(t.eval(i))
            val_loss = np.array(val_loss).mean()
            print(f"validation loss for step {step}: {val_loss}")

            wandb.log({'val/loss': val_loss}, step)

        step += 1

    ray.shutdown()
    delete_tpu(tpu_name, region)
