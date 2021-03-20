import argparse
import functools
import multiprocessing
import time

# from tensorboardX import SummaryWriter
import wandb

import numpy as np
import optax
import ray

# from enwik8_loader import TextLoader
from tqdm import tqdm

from tfrecord_loader import TFRecordNewInputs

from mesh_transformer import util
from mesh_transformer.TPU_cluster import TPUCluster
from mesh_transformer.transformer_shard import CausalTransformer
from ray_tpu import start_ray, get_connection, create_tpu, wait_til, delete_tpu

head_info = ray.init(dashboard_host="0.0.0.0")
address = head_info['redis_address']

tpu_name = "mesh-transformer-test-0"
bucket = "neo-models"
model_dir = "mesh_jax_owt"
gradient_accumulation_steps = 1
per_replica_batch = 8
tpus_per_replica = 8
tpu_size = 32
clean_start = False
val_batches = 100

assert tpu_size in [8, 32, 128, 256, 512]

# delete_tpu(tpu_name, "europe-west4-a")
create_tpu(tpu_name, "europe-west4-a", f"v3-{tpu_size}", True)
assert wait_til(tpu_name, "europe-west4-a", {'state': 'READY', 'health': 'HEALTHY'})

conns = get_connection(tpu_name, "europe-west4-a")

with multiprocessing.Pool(processes=len(conns)) as p:
     p.map(functools.partial(start_ray, address=address), conns)


# train_dataset = TextLoader("data/enwik8",
#                            batch_size=dataloader_batch,
#                            sample_size=1024, length=90000000)

opt = optax.chain(
    optax.scale(1/gradient_accumulation_steps),
    optax.clip_by_global_norm(1),
    optax.scale_by_adam(eps=1e-4),
    optax.scale(-1),
    optax.scale_by_schedule(util.gpt3_schedule(1_000, 20_000, 1e-4, 1e-5))
)

model_fn = functools.partial(CausalTransformer, dim=4096, heads=32, layer_count=1, vocab=50400, seq=1024, optimizer=opt)

t = TPUCluster((tpu_size//tpus_per_replica, tpus_per_replica), len(conns), model_fn)
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


train_dataset = TFRecordNewInputs("data/openwebtext2_new_inputs.train.index",
                                  batch_size=(gradient_accumulation_steps, per_replica_batch * tpu_size // tpus_per_replica),
                                  sample_size=1024,
                                  restore_state=train_load_restore)

val_dataset = TFRecordNewInputs("data/openwebtext2_new_inputs.val.index",
                                batch_size=(per_replica_batch * tpu_size // tpus_per_replica, ),
                                sample_size=1024)

start = time.time()
t.train(train_dataset.get_samples())
print(f"Train fn in {time.time() - start:.06}s")

start = time.time()
t.eval(val_dataset.get_samples())
print(f"Eval fn compiled in {time.time() - start:.06}s")

# writer = SummaryWriter(flush_secs=5)
wandb.init(project='mesh-transformer-jax', entity="eleutherai")
cfg = wandb.config
cfg.tpu_name = tpu_name
cfg.gradient_accumulation_steps = gradient_accumulation_steps
cfg.per_replica_batch = per_replica_batch
cfg.tpus_per_replica = tpus_per_replica
cfg.tpu_size = tpu_size

while True:
    loss = t.train(train_dataset.get_samples())
    wandb.log({'train/loss': loss}, step)

    if step % 100 == 0 and step:
        val_loss = []
        for i, _ in tqdm(zip(val_dataset.sample_once(), range(val_batches)), desc=f"validation for step {step}", total=val_batches):
            val_loss.append(t.eval(i))

        val_loss = np.array(val_loss).mean()

        print(f"validation loss for step {step}: {val_loss}")

        wandb.log({'val/loss': val_loss}, step)

        t.save(step, bucket, model_dir, aux={"train_loader": train_dataset.get_state()}, init=False)

    step += 1

ray.shutdown()
