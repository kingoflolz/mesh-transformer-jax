import argparse
import functools
import json
import multiprocessing
import time

import optax
import ray

from mesh_transformer import util
from mesh_transformer.TPU_cluster import TPUCluster
from mesh_transformer.transformer_shard import CausalTransformer
from mesh_transformer.util import clip_by_global_norm
from ray_tpu import start_ray, get_connection, create_tpu, wait_til
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
    # the same as train, but doesn't waste time doing stuff other than just getting step/s

    args = parse_args()
    params = json.load(open(args.config))

    tpu_name = args.tpu
    region = args.tpu_region
    preemptible = args.preemptible

    bucket = params["bucket"]
    model_dir = params["model_dir"]

    gradient_accumulation_steps = params.get("gradient_accumulation_steps", 1)
    per_replica_batch = params["per_replica_batch"]
    cores_per_replica = params["cores_per_replica"]
    tpu_size = params["tpu_size"]

    layers = params["layers"]
    d_model = params["d_model"]
    n_heads = params["n_heads"]
    n_vocab = params["n_vocab"]
    seq = params["seq"]
    norm = params["norm"]

    assert tpu_size in [8, 32, 128, 256, 512]

    create_tpu(tpu_name, region, f"v3-{tpu_size}", preemptible)
    assert wait_til(tpu_name, region, {'state': 'READY', 'health': 'HEALTHY'})

    conns = get_connection(tpu_name, region)

    assert len(conns) * 8 == tpu_size, "wrong size TPU for config"

    head_info = ray.init(dashboard_host="0.0.0.0")
    address = head_info['redis_address']

    with multiprocessing.Pool(processes=len(conns)) as p:
        p.map(functools.partial(start_ray, address=address), conns)

    opt = optax.chain(
        optax.scale(1 / gradient_accumulation_steps),
        clip_by_global_norm(1),
        optax.scale_by_adam(),
        optax.additive_weight_decay(0.1),
        optax.scale(-1),
        optax.scale_by_schedule(util.gpt3_schedule(1000, 100000, 1e-4, 1e-5))
    )

    params["optimizer"] = opt

    model_fn = functools.partial(CausalTransformer, params)

    t = TPUCluster((tpu_size // cores_per_replica, cores_per_replica), len(conns), model_fn)
    train_dataset = TFRecordNewInputs(f"data/{params['train_set']}",
                                      batch_size=(
                                      gradient_accumulation_steps, per_replica_batch * tpu_size // cores_per_replica),
                                      sample_size=params['seq'])

    val_dataset = TFRecordNewInputs(f"data/{params['val_set']}",
                                    batch_size=(per_replica_batch * tpu_size // cores_per_replica,),
                                    sample_size=params['seq'])

    param_count = t.param_count

    start = time.time()
    t.train(train_dataset.get_samples())
    print(f"Train fn compiled in {time.time() - start:.06}s")

    start = time.time()

    it = 16

    for i in range(it):
        iter_start = time.time()
        loss, last_loss = t.train(train_dataset.get_samples())
        print(f"Iter {i} done in {time.time() - iter_start:.06}s")

    total_time = time.time() - start

    print(f"100 iters done in {total_time:.06}s, avg {total_time/it:.06}s, {it/total_time:.06}/s")

    global_batch = gradient_accumulation_steps * per_replica_batch * tpu_size // cores_per_replica
    weight_flops = global_batch * seq * it * param_count
    attn_flops = global_batch * (seq ** 2) * it * 32 * 5120 * 16
    print(f"effective flops (not including attn): {weight_flops * 6 / total_time:.06}")
    print(f"MXU flops: {(weight_flops * 8 + attn_flops) / total_time:.06}")

    ray.shutdown()
