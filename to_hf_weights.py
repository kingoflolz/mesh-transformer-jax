####
# Script requires additional install of `pathy`
# run 'python to_hf_weights.py --help' to see usage.
####
# python to_hf_weights.py --input_ckpt /step_383500 --output_path resharded/debug_ckpt --cpu
####

import argparse
import io
import multiprocessing
import time
import warnings
import os
import re
from typing import Iterable, List, Tuple, Union

import jax
import jax.numpy as jnp
from jax.experimental import maps
import numpy as np
import optax
import torch

from tqdm import tqdm

from mesh_transformer.transformer_shard import CausalTransformer

try:
    from pathy import FluidPath, Pathy
except ModuleNotFoundError:
    raise ModuleNotFoundError(f"{__file__} requires `pathy`. Please run `pip install pathy`")

# xla: tell jax to not pre allocate all device memory
# and only allocate memory as needed.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

DEBUG = False

parser = argparse.ArgumentParser(
    description=(
        "Used to turn a sharded trained gpt-j checkpoint into pytorch hugging face format."
        "This script works best on a slimmed checkpoint (full checkpoints can be used but require ~100gb of ram)."
        "Currently, weights must be split into 8 shards for this to work."
        "All paths can be local or google cloud storage paths. S3 paths supported as well with `pip install pathy[s3]`."
        "Can be run on tpu, or on gpu with `pip install --upgrade jax==0.2.12 jaxlib==0.1.67+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html`"
    )
)
parser.add_argument(
    "--input_ckpt",
    metavar="path",
    type=str,
    help='path to model checkpoint folder. Google storage can be used with "gs://bucket/path/step_{n}" format.',
    required=True,
)
parser.add_argument(
    "--output_path",
    required=True,
    type=str,
    help='Full path to save checkpoint to. Google storage can be used with "gs://bucket/path" format.',
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Verbose printing.",
)
parser.add_argument(
    "--cpu",
    action="store_true",
    help="Run resharding on cpu instead of searching for jax device (i.e. gpu/tpu). Will default to cpu if jax wasn't installed with `+cuda110` option",
)
parser.add_argument(
    "--dtype",
    type=str,
    default="fp16",
    help="One of fp32, fp16 or bf16. Default=fp16. WARNING: Experimental. Make sure to check weights after conversion to make sure dtype information is retained.",
)


def process_args(
    input_ckpt: Union[FluidPath, str],
    output_path: Union[FluidPath, str],
    dtype: str = "fp16",
    cpu: bool = False,
    **kwargs,
):
    # validate paths and turn them into Pathy paths.
    input_ckpt = Pathy.fluid(str(input_ckpt))
    assert input_ckpt.is_dir(), f'no such directory "{input_ckpt}"'
    first_shard = input_ckpt / "shard_0"
    assert first_shard.is_dir(), f'no shards found at "{input_ckpt}"'

    output_path = Pathy.fluid(str(output_path))
    output_path.mkdir(exist_ok=True)

    # make sure dtype is valid
    assert dtype in {"fp16", "fp32", "bf16"}
    np_dtype = np.float16
    torch_dtype = torch.float16
    if dtype != "fp16":
        warnings.warn(
            "WARNING: Dtype support other than fp16 is Experimental. Make sure to check weights after conversion to make sure dtype information is retained."
        )
        if dtype == "bf16":
            # np doesn't have bfloat16 so float32 is used to retain information before converting to torch.
            np_dtype = np.float32
            torch_dtype = torch.bfloat16
        elif dtype == "fp32":
            np_dtype = np.float32
            torch_dtype = torch.float32

    # tell jax to run on cpu instead of gpu/tpu
    if cpu:
        jax.config.update("jax_platform_name", "cpu")

    return input_ckpt, output_path, np_dtype, torch_dtype


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


def get_tree_leaves_names_reduced(pytree) -> List[str]:

    leaves_ids = tree_leaves_with_names(pytree, to_id=id)
    leaves = jax.tree_leaves(pytree)
    return [leaves_ids[id(l)] for l in leaves]


layer_2_hf_inner_module_id = {
    "linear": "attn.attention.q_proj",
    "linear_1": "attn.attention.v_proj",
    "linear_2": "attn.attention.k_proj",
    "linear_3": "attn.attention.out_proj",
    "linear_4": "mlp.c_fc",
    "linear_5": "mlp.c_proj",
    "replicated_layer_norm": "ln_1",
}

projection_layer_2_hf_id_start = {
    "linear": "lm_head",
    "replicated_layer_norm": "transformer.ln_f",
}


def leave_name_to_hf_layer_id(leaf_name: str):
    if not leaf_name.startswith("/params"):
        if leaf_name == "/step":
            return None
        else:
            raise NotImplementedError(f"Unknown leaf name: {leaf_name}")

    match = re.search(
        r"\/params\/causal_transformer_shard\/~\/(?P<module_name>.*)\/~\/(?P<layer_name>.*)\/(?P<wb>.*)",
        leaf_name,
    )

    assert match, f'couldn\'t match pattern against: "{leaf_name}"'

    layer_name = match["layer_name"]
    module_name = match["module_name"]
    wb = match["wb"]

    if wb in {"w", "scale"}:
        weight_or_bias = "weight"
    elif wb in {"b", "offset"}:
        weight_or_bias = "bias"
    else:
        raise NotImplementedError(f"unknown weight/bais type identifier \"{wb}\" at end of: '{leaf_name}'")

    # switch statement based on top level module name
    if module_name == "embedding_shard":
        hf_id = f"transformer.wte.{weight_or_bias}"

    elif module_name.startswith("layer"):
        module_index = int(module_name.split("_")[-1])
        hf_inner_module_id = layer_2_hf_inner_module_id[layer_name]
        hf_id = f"transformer.h.{module_index}.{hf_inner_module_id}.{weight_or_bias}"

    elif module_name == "projection_shard":
        hf_id = f"{projection_layer_2_hf_id_start[layer_name]}.{weight_or_bias}"

    else:
        raise NotImplementedError(f"unknown leaf module type \"{module_name}\" in: '{leaf_name}'")

    if DEBUG:
        print(f"{leaf_name} \n\t -> {hf_id}")

    return hf_id


# TODO(nijkamp): rewrite this mess
def reshard(x, old_shape, do_shard_ln, do_shard_bias):
    # reshards using numpy arrays so as to not fill up jax memory
    if len(x.shape) == 1:
        out = np.array(x[0:1])

    elif len(x.shape) == 2:
        if do_shard_ln:
            out = np.array(x[0:1])
        elif do_shard_bias:
            out = np.reshape(np.sum(x, axis=0), old_shape)
        else:
            out = x.reshape(old_shape)

    elif len(x.shape) == 3:
        if x.shape[0] * x.shape[2] == old_shape[2]:
            out = np.transpose(x, (1, 0, 2)).reshape(old_shape)
        elif x.shape[0] * x.shape[1] == old_shape[1]:
            out = np.reshape(x, old_shape)
        else:
            raise NotImplementedError(f"unimplemented, {x.shape}, {old_shape}")
    else:
        raise NotImplementedError(f"unimplemented, {x}")
    return out


def read_npz(fpath: FluidPath):
    # read npz file of ndarrays
    with fpath.open("rb") as f:
        buf = f.read()
        f_io = io.BytesIO(buf)
        deserialized = np.load(
            f_io,
        )
        assert isinstance(deserialized, np.lib.npyio.NpzFile), f"Not an npz file {type(deserialized)=} {f=}"
        # arrays are only loaded when accessed. So we need to access them before returning
        arrays = []
        for i in deserialized:
            arr = deserialized[i]
            assert isinstance(arr, np.ndarray), f"Not a np.ndarray {type(arr)=} {f=}"
            arrays.append(arr)
        return arrays


def read_file_shards(ckpt_dir: FluidPath, fname: str, shards_in: int) -> List[List[np.ndarray]]:
    # read same file like "12.npz" accross all shard directories
    with multiprocessing.pool.ThreadPool(shards_in) as p:
        return list(
            p.imap(
                read_npz,
                [ckpt_dir / f"shard_{i}" / fname for i in range(shards_in)],
            )
        )


def lazy_read_ckpt_shards(ckpt_dir: FluidPath, shards_in: int, pieces: int = 16, reverse: bool = True):
    for i in range(pieces):
        # iterate through files in direction of choice
        fname = f"{(pieces-1) - i}.npz" if reverse else f"{i}.npz"
        if DEBUG:
            print(f"reading from {fname}")
        file_shards = read_file_shards(ckpt_dir, fname, shards_in)

        # iterate over layers in file returning all shards for each
        file_shards = list(zip(*file_shards))
        if reverse:
            file_shards = reversed(file_shards)
        yield from file_shards


def save_hf_layer(
    params: torch.Tensor, hf_layer_id: str, pt_save_idx: int, output_path: FluidPath, layer_map: dict
) -> Tuple[int, dict]:
    # Save layer as pt file and update layer mapping with the file name
    fname = f"b{pt_save_idx}.pt"
    save_loc = output_path / fname
    # add file to mapping of layer_ids to file names
    layer_map[hf_layer_id] = fname
    torch.save(params, save_loc.open(mode="wb"))

    # return incremented save index and updated layer_map
    return pt_save_idx + 1, layer_map


def unshard_leave(
    leave_shards: Iterable[np.ndarray], leave_name: str, old_shape: List[int], np_dtype=np.float16
):
    # reshard all leave shards into single shard.

    # stack leave shards into single np.ndarray
    x = np.stack(leave_shards)
    assert isinstance(x, jnp.ndarray)

    # As far as i can tell, this just re labels the dtype of arrays
    # labeled with "V2" dtype. In theory, V2 was just an alias for bfloat16
    # which needs to be relabeled in order for it to be understood.
    if x.dtype == np.dtype("V2"):
        x.dtype = jnp.bfloat16

    if DEBUG:
        print(f"RESHARDING: {leave_name=} {x.shape=} {old_shape=}")  # type: ignore

    # transform sharded array to match old_shape
    x = reshard(
        x,
        old_shape,
        do_shard_bias=leave_name.endswith("embedding_shard/~/linear/b") or leave_name.endswith("linear_5/b"),
        do_shard_ln=leave_name.endswith("replicated_layer_norm/offset")
        or leave_name.endswith("replicated_layer_norm/scale"),
    )
    assert x.shape == old_shape, f"Incompatible checkpoints {x.shape} vs {old_shape} {leave_name}"
    return x.astype(np_dtype)


def save_pytree_as_hf(
    pytree,
    input_ckpt: FluidPath,
    shards_in: int,
    output_path: FluidPath,
    n_layers: int = 28,
    np_dtype: type = np.float16,
    torch_dtype: torch.dtype = torch.float16,
):
    # Loads layers and names in reverse order to avoid loading unneeded opt_state layers
    # that are at the front of full (i.e. not slim) models.

    old_leave_shapes = [old.shape for old in jax.tree_flatten(pytree)[0]]
    leave_names = get_tree_leaves_names_reduced(pytree)
    del pytree

    assert len(old_leave_shapes) == len(leave_names), f"{len(old_leave_shapes)=}  {len(leave_names)=}"
    # get generator that emits all shards of leaves from npz files in reverse order
    loaded_shards_in = lazy_read_ckpt_shards(input_ckpt, shards_in, reverse=True)

    print("Reading and transforming layers/shards. This may take a while.")

    pt_save_idx = 0
    save_map = {}  # maps hf layer id's to .pt file names
    wte_first = None  # saves first instance of a wte weight in order to combine it with the second.
    # Reverse iteration to grab leave_names and old leaves from the back
    for i in tqdm(
        reversed(range(len(leave_names))), desc="Reading/Transforming Layers", total=len(leave_names)
    ):

        # load next shard with correstponding leave name and old shape
        x = next(loaded_shards_in)
        leave_name = leave_names[i]
        old_shape = old_leave_shapes[i]
        hf_layer_id = leave_name_to_hf_layer_id(leave_name)

        # If leave is not needed in hf model (/step')
        if not hf_layer_id:
            continue

        x = unshard_leave(x, leave_name, old_shape, np_dtype=np_dtype)
        # remove first empty dimension and transpose.
        x = torch.tensor(x.squeeze(0), dtype=torch_dtype).T

        # wte embedding weights/bias need to be combined since hf model has no wte.embedding.bias
        if hf_layer_id.startswith("transformer.wte"):
            # un/re-transpose since wte weight is only leave that shouldn't be transposed
            x = x.T
            # store first weight/bias then skip saving
            if wte_first is None:
                wte_first = x
                continue
            # combine second wte bias/weight with first then move on to saving with weight name
            else:
                x = x + wte_first
                hf_layer_id = "transformer.wte.weight"

        # save params as single file with proper hf id mapped to them in save_map
        pt_save_idx, save_map = save_hf_layer(x, hf_layer_id, pt_save_idx, output_path, save_map)

    # add attention bias layers
    attn_bias_weights = torch.tril(torch.tensor(np.ones((1, 1, 2048, 2048)), dtype=torch_dtype))
    attn_masked_bias_weights = torch.tensor(np.array(-1e9), dtype=torch_dtype)

    for i in range(n_layers):
        bias_id = f"transformer.h.{i}.attn.attention.bias"
        masked_bias_id = f"transformer.h.{i}.attn.attention.masked_bias"
        pt_save_idx, save_map = save_hf_layer(attn_bias_weights, bias_id, pt_save_idx, output_path, save_map)
        pt_save_idx, save_map = save_hf_layer(
            attn_masked_bias_weights, masked_bias_id, pt_save_idx, output_path, save_map
        )

    torch.save(save_map, (output_path / "m.pt").open(mode="wb"))


def save_sharded_to_hf_format(
    input_ckpt: Union[FluidPath, str],
    output_path: Union[FluidPath, str],
    cpu: bool = False,
    dtype: str = "fp16",
):

    input_ckpt, output_path, np_dtype, torch_dtype = process_args(
        input_ckpt=input_ckpt, output_path=output_path, dtype=dtype, cpu=cpu
    )

    params = {
        "layers": 28,
        "d_model": 4096,
        "n_heads": 16,
        "n_vocab": 50400,
        "norm": "layernorm",
        "pe": "rotary",
        "pe_rotary_dims": 64,
        "early_cast": True,
        "seq": 2048,
        "cores_per_replica": 1,
        "per_replica_batch": 1,
        "optimizer": optax.scale(0),
        "sampler": None,
    }

    devices = np.array([jax.devices()[0]]).reshape((1, 1))
    with maps.mesh(devices, ("dp", "mp")):
        network = CausalTransformer(params)

        save_pytree_as_hf(
            network.state,
            input_ckpt=input_ckpt,
            shards_in=8,
            output_path=output_path,
            n_layers=params["layers"],
            np_dtype=np_dtype,
            torch_dtype=torch_dtype,
        )


if __name__ == "__main__":
    args = vars(parser.parse_args())

    DEBUG = args["debug"]
    start = time.time()
    save_sharded_to_hf_format(args["input_ckpt"], args["output_path"], args["cpu"], args["dtype"])
    print(f"HF weights created in {(time.time() - start):.0f}s \"{args['output_path']}\"")