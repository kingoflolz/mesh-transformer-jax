import functools
import io
import time
from pathlib import Path

import jax
import pickle
import jax.numpy as jnp
import numpy as np
import multiprocessing

import cloudpickle
from tqdm import tqdm
from smart_open import open

pieces = 16 # how many files to split each shard across

@functools.partial(jax.jit, backend="cpu")
def index_weights(weights, idx):
    cpu_device = jax.devices("cpu")[0]
    return jax.device_put(jax.tree_map(lambda i: i[idx], weights), cpu_device)


def write(x, ckpt_dir):
    # start = time.time()
    idx, i = x
    file_path = ckpt_dir + f"{idx}.npz"
    with open(file_path, "wb") as f:
        np.savez(f, *i)
        # cloudpickle.dump(i, f)
        # print(f"written {idx} in {time.time() - start:.06}s")


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def write_ckpt(pytree, dir, shard):
    # ckpt_dir = Path(dir)
    # ckpt_dir.mkdir(parents=True, exist_ok=True)

    flattened, structure = jax.tree_flatten(pytree)

    start = time.time()
    # cpu_flattened = jax.device_put(flattened, cpu_device)
    cpu_flattened = index_weights(flattened, shard)
    # print(f"Moved indexed in {time.time() - start:.06}s")

    cpu_flattened_chunked = split(cpu_flattened, pieces)

    # start = time.time()
    # cpu_float = move_weights(cpu_flattened)
    # print(f"changed weight types in {time.time() - start:.06}s")

    with multiprocessing.pool.ThreadPool(pieces) as p:
        write_fn = functools.partial(write, ckpt_dir=f"{dir}shard_{shard}/")

        start = time.time()
        list((p.imap_unordered(write_fn, enumerate(cpu_flattened_chunked))))
        # print(f"written to gcs in {time.time() - start:.06}s")


def read_shard(ckpt_dir):
    out = []
    for idx in range(16):
        file_path = ckpt_dir + f"{idx}.npz"
        with open(file_path, "rb") as f:
            buf = f.read()
            f_io = io.BytesIO(buf)
            deserialized = np.load(f_io)
            for i in deserialized:
                out.append(deserialized[i])
    return out


def read_ckpt(pytree, dir, total_shards):
    old_flattened, structure = jax.tree_flatten(pytree)

    # TODO: figure out how to use a process pool here for more speed
    with multiprocessing.pool.ThreadPool(total_shards) as p:
        start = time.time()
        shards = list((p.imap(read_shard, [f"{dir}shard_{i}/" for i in range(total_shards)])))
        print(f"read from gcs in {time.time() - start:.06}s")

        unsharded = []

        for all_shards in zip(*shards):
            x = np.stack(all_shards)
            # No idea why this is V2...?
            if x.dtype == np.dtype('V2'):
                x.dtype = jnp.bfloat16
            unsharded.append(x)

        for new, old in zip(unsharded, old_flattened):
            assert new.shape == old.shape, f"Incompatible checkpoints {new.shape} vs {old.shape}"
            # assert new.dtype == old.dtype, f"Incompatible checkpoints {new.dtype} vs {old.dtype}"

    return jax.tree_unflatten(structure, unsharded)
