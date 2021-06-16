import argparse
import json
import time

import jax
import numpy as np
import optax

from mesh_transformer import util
from mesh_transformer.checkpoint import read_ckpt, write_ckpt
from mesh_transformer.transformer_shard import CausalTransformer
from smart_open import open

from mesh_transformer.util import clip_by_global_norm, to_bf16, to_f16


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Config file location")
    parser.add_argument("--ckpt-step", type=int, default=-1, help="Step number of the checkpoint to convert (if not specified, converts the most recent checkpoint)")
    parser.add_argument("--f16", default=False, action="store_true", help="Convert to float16 (instead of bfloat16)")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    params = json.load(open(args.config))
    convert_fn = to_f16 if args.f16 else to_bf16

    cores_per_replica = params["cores_per_replica"]

    assert cores_per_replica <= 8

    bucket = params["bucket"]
    model_dir = params["model_dir"]

    params["optimizer"] = optax.chain(
        optax.scale(1),
        clip_by_global_norm(1),
        optax.scale_by_adam(),
        optax.additive_weight_decay(0),
        optax.scale(-1),
        optax.scale_by_schedule(util.gpt3_schedule(0, 1, 0, 0))
    )

    start = time.time()
    print(f"jax devices: {jax.device_count()}")
    print(f"jax runtime initialized in {time.time() - start:.06}s")

    mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
    devices = np.array(jax.devices()).reshape(mesh_shape)

    with open(f"gs://{bucket}/{model_dir}/meta.json", "r") as f:
        meta = json.load(f)

    if args.ckpt_step > -1:
        ckpt_step = args.ckpt_step
    else:
        ckpt_step = meta["checkpoints"][-1]
    print(f"using checkpoint {ckpt_step}")

    with jax.experimental.maps.mesh(devices, ('dp', 'mp')):
        network = CausalTransformer(params)

        start = time.time()
        network.state = read_ckpt(network.state, f"gs://{bucket}/{model_dir}/step_{ckpt_step}/", devices.shape[1])
        print(f"network loaded in {time.time() - start:.06}s")

        start = time.time()
        del network.state["opt_state"]

        network.state["params"] = convert_fn(network.state["params"])
        print(f"network converted in {time.time() - start:.06}s")

        suffix = "_slim_f16" if args.f16 else "_slim"

        for i in range(cores_per_replica):
            write_ckpt(network.state, f"gs://{bucket}/{model_dir}{suffix}/step_{ckpt_step}/", i)
            print(f"written shard {i}")
