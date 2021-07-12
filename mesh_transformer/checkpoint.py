import functools
import io
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import multiprocessing

from smart_open import open

pieces = 16  # how many files to split each shard across


@functools.partial(jax.jit, backend="cpu")
def index_weights(weights, idx):
    cpu_device = jax.devices("cpu")[0]
    return jax.device_put(jax.tree_map(lambda i: i[idx], weights), cpu_device)


def write(x, ckpt_dir):
    # start = time.time()
    idx, i = x
    file_path = ckpt_dir + f"{idx}.npz"
    for _ in range(3):
        try:
            with open(file_path, "wb") as f:
                np.savez(f, *i)
                # cloudpickle.dump(i, f)
                # print(f"written {idx} in {time.time() - start:.06}s")
            return
        except:
            print("save failed, trying again")

    print("save failed 3 times, exiting")
    raise Exception("save failed")


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


def reshard(x, old_shape):
    if len(x.shape) == 1:
        # print("epoch")
        # print(x)
        out = x[0:1]

    elif len(x.shape) == 2:
        # print(f"LN/bias {x.shape}")
        # print(x[:, :16])

        if (x[1:] == x[-1]).all():
            # print("LN")
            if (x[1:] == 0).all() or (x[1:] == 1).all():
                out = x[0:1]
            else:
                # print("shard bias")
                out = x[0:1] * x.shape[0] / old_shape[0]
        else:
            # print("bias")
            out = x.reshape(old_shape)

        print(out[:, :16])

    elif len(x.shape) == 3:
        # print(f"weight {x.shape}")
        if x.shape[0] * x.shape[2] == old_shape[2]:
            # print("case 1")
            out = jnp.transpose(x, (1, 0, 2)).reshape(old_shape)
        elif x.shape[0] * x.shape[1] == old_shape[1]:
            # print("case 2")
            out = x.reshape(old_shape)
        else:
            raise Exception(f"unimplemented, {x.shape}, {old_shape}")
    else:
        raise Exception(f"unimplemented, {x}")

    return out


def read_ckpt(pytree, dir, shards_in, shards_out=None, load_opt=True):
    if shards_out is None:
        shards_out = shards_in

    old_flattened, structure = jax.tree_flatten(pytree)

    original_opt_state = pytree["opt_state"]

    # TODO: figure out how to use a process pool here for more speed
    with multiprocessing.pool.ThreadPool(shards_in) as p:
        start = time.time()
        shards = list((p.imap(read_shard, [f"{dir}shard_{i}/" for i in range(shards_in)])))
        print(f"read from disk/gcs in {time.time() - start:.06}s")

        unsharded = []

        for old, *all_shards in zip(old_flattened, *shards):
            x = np.stack(all_shards)
            # No idea why this is V2...?
            if x.dtype == np.dtype('V2'):
                x.dtype = jnp.bfloat16

            if shards_out != shards_in:
                x = reshard(x, old.shape)
            unsharded.append(x)

            assert x.shape == old.shape, f"Incompatible checkpoints {x.shape} vs {old.shape}"

    loaded_pytree = jax.tree_unflatten(structure, unsharded)

    if not load_opt:
        loaded_pytree['opt_state'] = original_opt_state
    return loaded_pytree


def parallel_write(arrays, fname):
    with open(fname, "wb") as f:
        np.savez(f, *arrays)


def parallel_read(old, fname):
    old_val, treedef = jax.tree_flatten(old)
    with open(fname, "rb") as f:
        buf = f.read()
        f_io = io.BytesIO(buf)
        loaded = np.load(f_io)

    new_vals = []
    for i in loaded:
        new_vals.append(loaded[i])

    for o, n in zip(new_vals, old_val):
        assert o.shape == n.shape, "Incompatible checkpoint"

        if o.dtype == np.dtype('V2'):
            o.dtype = jnp.bfloat16

    return jax.tree_unflatten(treedef, new_vals)


def write_ckpt_v2(model_state, dir):
    start = time.time()
    if jax.host_id() == 0:
        print("step:", model_state["step"])
        with open(dir + "/meta.json", "w") as f:
            json.dump({"total_hosts": jax.host_count(), "step": int(model_state["step"])}, f)
        print(f"meta written in {time.time() - start:.06}s")

    start = time.time()
    parallel_write(jax.tree_flatten(model_state["params"])[0], dir + f"/params/shard_{jax.host_id()}.npz")
    print(f"params written in {time.time() - start:.06}s")

    start = time.time()
    parallel_write(jax.tree_flatten(model_state["opt_state"])[0], dir + f"/opt_state/shard_{jax.host_id()}.npz")
    print(f"opt_state written in {time.time() - start:.06}s")


def load_ckpt_v2(model_state, dir):
    start = time.time()
    with open(dir + "/meta.json", "r") as f:
        meta = json.load(f)

    # TODO: make this work in the general case
    assert meta["total_hosts"] == jax.host_count(), "Must load with same number of hosts as when saved"

    print(f"meta loaded in {time.time() - start:.06}s")

    new_model_state = {
        "step": np.array(meta["step"]),
    }

    start = time.time()
    new_model_state["params"] = parallel_read(model_state["params"], dir + f"/params/shard_{jax.host_id()}.npz")
    print(f"params loaded in {time.time() - start:.06}s")

    start = time.time()
    new_model_state["opt_state"] = parallel_read(model_state["opt_state"], dir + f"/opt_state/shard_{jax.host_id()}.npz")
    print(f"opt_state loaded in {time.time() - start:.06}s")

    return new_model_state
