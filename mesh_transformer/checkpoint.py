import functools
import io
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import multiprocessing

import ray
from smart_open import open

from mesh_transformer.util import head_print

pieces = 16  # how many files to split each shard across


def fix_dtype(pytree):
    def fix(x):
        if x.dtype == np.dtype('V2'):
            x.dtype = jnp.bfloat16
        return jnp.asarray(x)

    return jax.tree_map(fix, pytree)


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

    def _unshard(shards, old_flattened):
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
        return unsharded
    try:
        unsharded = _unshard(shards, old_flattened)
    except AssertionError:
        load_opt = False  # no opt to load in ckpt
        del pytree['opt_state']
        old_flattened, structure = jax.tree_flatten(pytree)
        unsharded = _unshard(shards, old_flattened)

    loaded_pytree = jax.tree_unflatten(structure, unsharded)

    if not load_opt:
        loaded_pytree['opt_state'] = original_opt_state
    return loaded_pytree


def read_ckpt_lowmem(pytree, dir, shards_in, shards_out=None, load_opt=True):
    if shards_out is None:
        shards_out = shards_in

    old_flattened, structure = jax.tree_flatten(pytree)

    original_opt_state = pytree["opt_state"]

    def _unshard():
        start = time.time()
        unsharded = []
        devices = jax.devices()
        device_count = len(devices)
        device_index = 0

        for file_index in range(pieces):
            array_keys = [*np.load(f"{dir}shard_0/{file_index}.npz").keys()]
            for array_index in range(len(array_keys)):
                unstacked = []
                for shard_index in range(shards_in):
                    npz = np.load(f"{dir}shard_{shard_index}/{file_index}.npz")
                    array = npz[array_keys[array_index]]
                    if array.dtype == 'V2':
                        array.dtype = jnp.bfloat16
                    unstacked.append(array)

                x = jax.device_put(jnp.stack(unstacked), device=devices[device_index % device_count])

                if shards_out != shards_in:
                    x = reshard(x, old_flattened[device_index].shape)
                unsharded.append(x)

                assert x.shape == old_flattened[device_index].shape, f"Incompatible checkpoints {x.shape} vs {old_flattened[device_index].shape}"
                device_index += 1

        print(f"read from disk/gcs in {time.time() - start:.06}s")
        return unsharded

    try:
        unsharded = _unshard()
    except AssertionError:
        load_opt = False  # no opt to load in ckpt
        del pytree['opt_state']
        old_flattened, structure = jax.tree_flatten(pytree)
        unsharded = _unshard()

    loaded_pytree = jax.tree_unflatten(structure, unsharded)

    if not load_opt:
        loaded_pytree['opt_state'] = original_opt_state
    return loaded_pytree


def parallel_write(arrays, fname):
    # TODO: make this actually parallel
    with open(fname, "wb") as f:
        np.savez(f, *arrays)


def parallel_read(old, fname, validate=True):
    old_vals, treedef = jax.tree_flatten(old)

    if "gs://" in fname:
        # TODO: make this actually parallel
        with open(fname, "rb") as f:
            buf = f.read()
            f_io = io.BytesIO(buf)
            loaded = np.load(f_io)
    else:
        loaded = np.load(fname, mmap_mode='r')

    new_vals = []
    for i in loaded:
        new_vals.append(loaded[i])

    assert len(new_vals) == len(old_vals), "Incompatible checkpoint"

    for o, n in zip(new_vals, old_vals):
        if validate:
            assert o.shape == n.shape, "Incompatible checkpoint"

    return jax.tree_unflatten(treedef, fix_dtype(new_vals))


def tree_flatten_with_names(pytree, is_leaf, path="", to_id=id):
    id_to_name = {}
    if getattr(pytree, "items", None):
        for k, v in pytree.items():
            k_path = f"{path}/{k}"
            if is_leaf(v):
                id_to_name[to_id(v)] = k_path
            else:
                id_to_name = {**id_to_name, **tree_flatten_with_names(v, is_leaf=is_leaf, path=k_path)}
    elif getattr(pytree, "__getitem__", None):
        for v in pytree:
            if is_leaf(v):
                id_to_name[to_id(v)] = path
            else:
                id_to_name = {**id_to_name, **tree_flatten_with_names(v, is_leaf=is_leaf, path=path)}
    else:
        id_to_name[to_id(pytree)] = path
    return id_to_name


def tree_leaves_with_names(pytree, to_id=id):
    leaves = jax.tree_leaves(pytree)
    is_leaf = lambda x: not isinstance(x, list) and to_id(x) in [to_id(x) for x in leaves]
    return tree_flatten_with_names(pytree, is_leaf)


def write_ckpt_v2(model_state, dir):
    start = time.time()
    if jax.host_id() == 0:
        param_map = tree_leaves_with_names(model_state["params"])
        opt_map = tree_leaves_with_names(model_state["opt_state"])

        meta = {
                    "total_hosts": jax.host_count(),
                    "step": int(model_state["step"]),
                    "param_order": [param_map[id(i)] for i in jax.tree_leaves(model_state["params"])],
                    "opt_order": [opt_map[id(i)] for i in jax.tree_leaves(model_state["opt_state"])]
        }

        print("step:", model_state["step"])
        with open(dir + "/meta.json", "w") as f:
            json.dump(meta, f)
        print(f"meta written in {time.time() - start:.06}s")

    start = time.time()
    parallel_write(jax.tree_flatten(model_state["params"])[0], dir + f"/params/shard_{jax.host_id()}.npz")
    head_print(f"params written in {time.time() - start:.06}s")

    start = time.time()
    parallel_write(jax.tree_flatten(model_state["opt_state"])[0], dir + f"/opt_state/shard_{jax.host_id()}.npz")
    head_print(f"opt_state written in {time.time() - start:.06}s")


def read_sharded_v2(state, dir, checkpoint_hosts, state_shard):
    files_per_host = checkpoint_hosts // jax.host_count()

    assert files_per_host >= 1, "can't restore model to larger pod than was trained on (yet)"
    assert jax.host_count() * files_per_host == checkpoint_hosts, "weird host count"

    if files_per_host == 1:
        head_print("using fast path of checkpoint restore (save shards == read shards)")
        parallel_read(state, dir + f"/shard_{jax.host_id()}.npz")

    @ray.remote
    def read_remote(old, fname):
        return parallel_read(old, fname, validate=False)

    start_idx = files_per_host * jax.host_id()

    skeleton = jax.tree_map(lambda x: jnp.zeros_like(x, shape=()), state)  # a full pytree just to carry dtypes

    refs = [
        read_remote.remote(skeleton, f"{dir}/shard_{i}.npz")
        for i in range(start_idx, start_idx + files_per_host)
    ]

    values = ray.get(refs)

    def all_array_equal(iterator):
        try:
            iterator = iter(iterator)
            first = next(iterator)
            return all(jnp.array_equal(first, rest) for rest in iterator)
        except StopIteration:
            return True

    def reshard_v2(old, shard_strategy, *new_values):
        rep_dim_count = shard_strategy.count(None)
        total_dim_count = len(shard_strategy)

        # head_print("old.shape", old.shape)
        # head_print("shard_strategy", shard_strategy)

        assert len(old.shape) == total_dim_count

        if rep_dim_count == total_dim_count:
            # fully replicated
            assert all_array_equal(new_values)
            return fix_dtype(new_values[0])

        shard_dim = [idx for idx, dim in enumerate(shard_strategy) if dim is not None and "mp" in dim]

        # only support sharding in 1d for now
        assert len(shard_dim) == 1
        shard_dim = shard_dim[0]

        ret_val = jnp.concatenate(fix_dtype(new_values), axis=shard_dim)
        assert old.shape == ret_val.shape

        return jax.device_put(ret_val, jax.devices("cpu")[0])

    # head_print("state", jax.tree_structure(state))
    # head_print("state_shard", jax.tree_structure(state_shard))
    # head_print("values", jax.tree_structure(values[0]))

    return jax.tree_multimap(reshard_v2, *([state, state_shard] + values))


def load_ckpt_v2(model_state, dir, state_shard, load_opt):
    start = time.time()
    with open(dir + "meta.json", "r") as f:
        meta = json.load(f)

    ckpt_hosts = meta["total_hosts"]

    head_print(f"meta loaded in {time.time() - start:.06}s")

    new_state = {
        "step": np.array([meta["step"]]),
    }

    start = time.time()
    new_state["params"] = read_sharded_v2(model_state["params"],
                                          dir + "params",
                                          ckpt_hosts,
                                          state_shard["params"])
    head_print(f"params loaded in {time.time() - start:.06}s")

    if not load_opt:
        return new_state

    start = time.time()
    new_state["opt_state"] = read_sharded_v2(model_state["opt_state"],
                                             dir + "opt_state",
                                             ckpt_hosts,
                                             state_shard["opt_state"])
    head_print(f"opt_state loaded in {time.time() - start:.06}s")

    return new_state
