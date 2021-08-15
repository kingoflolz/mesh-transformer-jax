####
# run with 'help' arg for usage.
####


"""
python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools
pip install -r requirements.txt
pip install pathy
pip install --upgrade jax==0.2.12 jaxlib==0.1.67+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html
"""
import os
import re
from typing import List, Tuple, Union

from jax._src.numpy.lax_numpy import ndarray

# xla: tells jax to not pre allocate all device memory
# and only allocate memory as needed.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import io
import multiprocessing
import torch
from tqdm import tqdm

import numpy as np
from pathy import Pathy, FluidPath

#! Some imports are done after argument processing so that cli is faster
# i.e. no waiting a minute for the `help` command or a missed arg


DEBUG = False


def process_args(
    input_ckpt: Union[FluidPath, str],
    output_path: Union[FluidPath, str],
    **kwargs,
):
    # validate paths and turn them into Pathy paths.
    # seperated from reshard_checkpoint so that args can be validated before expensive imports
    input_ckpt = Pathy.fluid(str(input_ckpt))
    assert input_ckpt.is_dir(), f'no such directory "{input_ckpt}"'
    first_shard = input_ckpt / "shard_0"
    assert first_shard.is_dir(), f'no shards found at "{input_ckpt}"'

    output_path = Pathy.fluid(str(output_path))
    output_path.mkdir(exist_ok=True)

    return input_ckpt, output_path


# parse args before importing expensive
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Used to turn a sharded trained gpt-j checkpoint into pytorch hugging face format."
            "This script works best on a slimmed checkpoint (full checkpoints can be used but require ~100gb of ram)."
            "Currently, weights must be split into 8 shards for this to work."
            "All paths can be local or google cloud storage paths. S3 paths supported as well with `pip install pathy[s3]`."
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
        help="Run resharding on cpu if not on v3-8 tpu.",
    )
    # TODO(dwarf): add support for configs?
    args = vars(parser.parse_args())
    # validate args
    process_args(**args)

    DEBUG = args["debug"]


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


def get_tree_leaves_names_original(params):

    params["optimizer"] = optax.chain(
        optax.scale(1),
        util.clip_by_global_norm(1),
        optax.scale_by_adam(),
        optax.additive_weight_decay(0),
        optax.scale(-1),
        optax.scale_by_schedule(util.gpt3_schedule(0, 1, 0, 0)),
    )

    devices = np.array([jax.devices()[0]]).reshape((1, 1))
    with jax.experimental.maps.mesh(devices, ("dp", "mp")):  # type: ignore
        network = CausalTransformer(params)
        leaves_ids = tree_leaves_with_names(network.state, to_id=id)
        leaves = jax.tree_leaves(network.state)
        leaves_names = [leaves_ids[id(l)] for l in leaves]

        return leaves_names


def get_tree_leaves_names_reduced(params):

    jax.config.update("jax_platform_name", "cpu")

    params["optimizer"] = optax.scale(0)

    devices = np.array([jax.devices()[0]]).reshape((1, 1))
    with jax.experimental.maps.mesh(devices, ("dp", "mp")):  # type: ignore
        network = CausalTransformer(params)
        leaves_ids = tree_leaves_with_names(network.state, to_id=id)
        leaves = jax.tree_leaves(network.state)
        leaves_names = [leaves_ids[id(l)] for l in leaves]

        return leaves_names


# This one is only used if checkpoint hasn't been slimmed
# TODO: is this needed? should it just require a slimmed checkpoint?
# leaves_names_original = get_tree_leaves_names_original(params)
# print(leaves_names_original)
leaves_names_original = [
    "/opt_state",
    "/opt_state/causal_transformer_shard/~/embedding_shard/~/linear/b",
    "/opt_state/causal_transformer_shard/~/embedding_shard/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_0/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_0/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_0/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_0/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_0/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_0/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_0/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_0/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_0/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_0/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_1/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_1/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_1/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_1/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_1/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_1/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_1/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_1/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_1/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_1/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_10/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_10/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_10/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_10/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_10/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_10/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_10/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_10/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_10/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_10/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_11/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_11/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_11/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_11/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_11/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_11/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_11/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_11/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_11/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_11/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_12/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_12/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_12/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_12/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_12/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_12/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_12/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_12/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_12/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_12/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_13/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_13/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_13/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_13/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_13/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_13/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_13/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_13/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_13/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_13/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_14/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_14/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_14/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_14/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_14/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_14/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_14/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_14/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_14/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_14/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_15/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_15/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_15/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_15/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_15/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_15/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_15/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_15/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_15/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_15/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_16/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_16/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_16/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_16/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_16/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_16/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_16/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_16/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_16/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_16/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_17/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_17/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_17/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_17/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_17/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_17/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_17/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_17/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_17/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_17/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_18/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_18/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_18/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_18/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_18/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_18/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_18/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_18/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_18/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_18/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_19/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_19/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_19/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_19/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_19/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_19/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_19/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_19/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_19/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_19/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_2/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_2/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_2/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_2/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_2/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_2/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_2/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_2/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_2/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_2/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_20/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_20/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_20/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_20/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_20/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_20/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_20/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_20/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_20/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_20/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_21/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_21/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_21/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_21/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_21/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_21/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_21/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_21/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_21/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_21/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_22/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_22/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_22/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_22/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_22/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_22/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_22/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_22/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_22/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_22/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_23/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_23/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_23/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_23/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_23/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_23/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_23/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_23/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_23/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_23/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_24/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_24/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_24/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_24/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_24/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_24/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_24/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_24/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_24/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_24/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_25/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_25/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_25/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_25/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_25/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_25/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_25/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_25/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_25/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_25/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_26/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_26/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_26/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_26/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_26/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_26/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_26/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_26/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_26/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_26/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_27/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_27/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_27/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_27/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_27/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_27/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_27/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_27/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_27/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_27/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_3/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_3/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_3/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_3/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_3/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_3/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_3/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_3/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_3/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_3/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_4/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_4/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_4/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_4/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_4/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_4/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_4/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_4/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_4/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_4/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_5/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_5/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_5/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_5/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_5/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_5/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_5/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_5/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_5/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_5/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_6/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_6/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_6/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_6/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_6/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_6/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_6/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_6/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_6/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_6/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_7/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_7/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_7/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_7/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_7/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_7/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_7/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_7/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_7/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_7/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_8/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_8/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_8/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_8/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_8/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_8/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_8/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_8/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_8/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_8/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_9/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_9/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_9/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_9/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_9/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_9/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_9/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_9/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_9/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_9/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/projection_shard/~/linear/b",
    "/opt_state/causal_transformer_shard/~/projection_shard/~/linear/w",
    "/opt_state/causal_transformer_shard/~/projection_shard/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/projection_shard/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/embedding_shard/~/linear/b",
    "/opt_state/causal_transformer_shard/~/embedding_shard/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_0/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_0/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_0/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_0/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_0/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_0/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_0/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_0/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_0/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_0/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_1/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_1/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_1/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_1/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_1/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_1/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_1/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_1/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_1/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_1/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_10/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_10/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_10/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_10/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_10/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_10/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_10/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_10/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_10/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_10/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_11/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_11/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_11/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_11/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_11/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_11/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_11/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_11/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_11/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_11/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_12/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_12/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_12/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_12/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_12/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_12/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_12/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_12/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_12/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_12/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_13/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_13/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_13/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_13/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_13/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_13/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_13/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_13/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_13/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_13/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_14/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_14/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_14/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_14/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_14/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_14/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_14/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_14/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_14/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_14/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_15/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_15/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_15/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_15/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_15/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_15/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_15/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_15/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_15/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_15/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_16/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_16/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_16/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_16/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_16/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_16/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_16/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_16/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_16/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_16/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_17/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_17/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_17/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_17/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_17/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_17/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_17/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_17/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_17/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_17/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_18/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_18/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_18/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_18/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_18/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_18/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_18/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_18/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_18/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_18/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_19/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_19/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_19/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_19/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_19/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_19/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_19/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_19/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_19/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_19/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_2/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_2/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_2/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_2/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_2/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_2/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_2/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_2/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_2/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_2/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_20/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_20/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_20/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_20/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_20/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_20/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_20/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_20/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_20/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_20/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_21/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_21/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_21/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_21/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_21/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_21/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_21/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_21/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_21/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_21/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_22/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_22/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_22/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_22/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_22/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_22/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_22/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_22/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_22/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_22/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_23/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_23/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_23/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_23/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_23/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_23/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_23/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_23/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_23/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_23/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_24/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_24/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_24/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_24/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_24/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_24/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_24/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_24/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_24/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_24/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_25/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_25/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_25/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_25/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_25/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_25/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_25/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_25/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_25/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_25/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_26/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_26/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_26/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_26/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_26/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_26/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_26/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_26/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_26/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_26/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_27/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_27/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_27/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_27/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_27/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_27/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_27/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_27/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_27/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_27/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_3/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_3/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_3/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_3/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_3/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_3/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_3/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_3/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_3/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_3/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_4/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_4/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_4/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_4/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_4/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_4/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_4/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_4/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_4/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_4/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_5/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_5/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_5/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_5/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_5/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_5/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_5/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_5/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_5/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_5/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_6/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_6/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_6/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_6/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_6/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_6/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_6/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_6/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_6/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_6/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_7/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_7/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_7/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_7/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_7/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_7/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_7/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_7/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_7/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_7/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_8/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_8/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_8/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_8/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_8/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_8/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_8/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_8/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_8/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_8/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/layer_9/~/linear/w",
    "/opt_state/causal_transformer_shard/~/layer_9/~/linear_1/w",
    "/opt_state/causal_transformer_shard/~/layer_9/~/linear_2/w",
    "/opt_state/causal_transformer_shard/~/layer_9/~/linear_3/w",
    "/opt_state/causal_transformer_shard/~/layer_9/~/linear_4/b",
    "/opt_state/causal_transformer_shard/~/layer_9/~/linear_4/w",
    "/opt_state/causal_transformer_shard/~/layer_9/~/linear_5/b",
    "/opt_state/causal_transformer_shard/~/layer_9/~/linear_5/w",
    "/opt_state/causal_transformer_shard/~/layer_9/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/layer_9/~/replicated_layer_norm/scale",
    "/opt_state/causal_transformer_shard/~/projection_shard/~/linear/b",
    "/opt_state/causal_transformer_shard/~/projection_shard/~/linear/w",
    "/opt_state/causal_transformer_shard/~/projection_shard/~/replicated_layer_norm/offset",
    "/opt_state/causal_transformer_shard/~/projection_shard/~/replicated_layer_norm/scale",
    "/opt_state",
    "/params/causal_transformer_shard/~/embedding_shard/~/linear/b",
    "/params/causal_transformer_shard/~/embedding_shard/~/linear/w",
    "/params/causal_transformer_shard/~/layer_0/~/linear/w",
    "/params/causal_transformer_shard/~/layer_0/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_0/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_0/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_0/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_0/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_0/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_0/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_0/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_0/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_1/~/linear/w",
    "/params/causal_transformer_shard/~/layer_1/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_1/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_1/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_1/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_1/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_1/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_1/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_1/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_1/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_10/~/linear/w",
    "/params/causal_transformer_shard/~/layer_10/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_10/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_10/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_10/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_10/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_10/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_10/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_10/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_10/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_11/~/linear/w",
    "/params/causal_transformer_shard/~/layer_11/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_11/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_11/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_11/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_11/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_11/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_11/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_11/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_11/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_12/~/linear/w",
    "/params/causal_transformer_shard/~/layer_12/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_12/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_12/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_12/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_12/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_12/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_12/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_12/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_12/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_13/~/linear/w",
    "/params/causal_transformer_shard/~/layer_13/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_13/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_13/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_13/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_13/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_13/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_13/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_13/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_13/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_14/~/linear/w",
    "/params/causal_transformer_shard/~/layer_14/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_14/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_14/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_14/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_14/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_14/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_14/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_14/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_14/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_15/~/linear/w",
    "/params/causal_transformer_shard/~/layer_15/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_15/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_15/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_15/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_15/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_15/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_15/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_15/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_15/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_16/~/linear/w",
    "/params/causal_transformer_shard/~/layer_16/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_16/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_16/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_16/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_16/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_16/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_16/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_16/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_16/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_17/~/linear/w",
    "/params/causal_transformer_shard/~/layer_17/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_17/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_17/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_17/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_17/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_17/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_17/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_17/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_17/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_18/~/linear/w",
    "/params/causal_transformer_shard/~/layer_18/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_18/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_18/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_18/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_18/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_18/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_18/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_18/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_18/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_19/~/linear/w",
    "/params/causal_transformer_shard/~/layer_19/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_19/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_19/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_19/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_19/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_19/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_19/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_19/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_19/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_2/~/linear/w",
    "/params/causal_transformer_shard/~/layer_2/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_2/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_2/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_2/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_2/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_2/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_2/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_2/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_2/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_20/~/linear/w",
    "/params/causal_transformer_shard/~/layer_20/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_20/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_20/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_20/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_20/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_20/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_20/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_20/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_20/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_21/~/linear/w",
    "/params/causal_transformer_shard/~/layer_21/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_21/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_21/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_21/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_21/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_21/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_21/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_21/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_21/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_22/~/linear/w",
    "/params/causal_transformer_shard/~/layer_22/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_22/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_22/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_22/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_22/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_22/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_22/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_22/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_22/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_23/~/linear/w",
    "/params/causal_transformer_shard/~/layer_23/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_23/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_23/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_23/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_23/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_23/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_23/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_23/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_23/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_24/~/linear/w",
    "/params/causal_transformer_shard/~/layer_24/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_24/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_24/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_24/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_24/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_24/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_24/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_24/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_24/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_25/~/linear/w",
    "/params/causal_transformer_shard/~/layer_25/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_25/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_25/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_25/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_25/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_25/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_25/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_25/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_25/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_26/~/linear/w",
    "/params/causal_transformer_shard/~/layer_26/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_26/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_26/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_26/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_26/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_26/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_26/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_26/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_26/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_27/~/linear/w",
    "/params/causal_transformer_shard/~/layer_27/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_27/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_27/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_27/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_27/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_27/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_27/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_27/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_27/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_3/~/linear/w",
    "/params/causal_transformer_shard/~/layer_3/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_3/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_3/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_3/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_3/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_3/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_3/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_3/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_3/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_4/~/linear/w",
    "/params/causal_transformer_shard/~/layer_4/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_4/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_4/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_4/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_4/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_4/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_4/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_4/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_4/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_5/~/linear/w",
    "/params/causal_transformer_shard/~/layer_5/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_5/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_5/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_5/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_5/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_5/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_5/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_5/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_5/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_6/~/linear/w",
    "/params/causal_transformer_shard/~/layer_6/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_6/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_6/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_6/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_6/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_6/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_6/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_6/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_6/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_7/~/linear/w",
    "/params/causal_transformer_shard/~/layer_7/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_7/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_7/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_7/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_7/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_7/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_7/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_7/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_7/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_8/~/linear/w",
    "/params/causal_transformer_shard/~/layer_8/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_8/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_8/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_8/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_8/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_8/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_8/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_8/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_8/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_9/~/linear/w",
    "/params/causal_transformer_shard/~/layer_9/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_9/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_9/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_9/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_9/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_9/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_9/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_9/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_9/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/projection_shard/~/linear/b",
    "/params/causal_transformer_shard/~/projection_shard/~/linear/w",
    "/params/causal_transformer_shard/~/projection_shard/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/projection_shard/~/replicated_layer_norm/scale",
    "/step",
]

# leaves_names_reduced = get_tree_leaves_names_reduced(params)
# print(leaves_names_reduced)
leaves_names_reduced = [
    "/params/causal_transformer_shard/~/embedding_shard/~/linear/b",
    "/params/causal_transformer_shard/~/embedding_shard/~/linear/w",
    "/params/causal_transformer_shard/~/layer_0/~/linear/w",
    "/params/causal_transformer_shard/~/layer_0/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_0/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_0/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_0/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_0/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_0/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_0/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_0/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_0/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_1/~/linear/w",
    "/params/causal_transformer_shard/~/layer_1/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_1/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_1/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_1/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_1/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_1/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_1/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_1/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_1/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_10/~/linear/w",
    "/params/causal_transformer_shard/~/layer_10/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_10/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_10/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_10/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_10/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_10/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_10/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_10/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_10/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_11/~/linear/w",
    "/params/causal_transformer_shard/~/layer_11/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_11/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_11/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_11/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_11/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_11/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_11/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_11/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_11/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_12/~/linear/w",
    "/params/causal_transformer_shard/~/layer_12/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_12/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_12/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_12/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_12/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_12/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_12/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_12/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_12/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_13/~/linear/w",
    "/params/causal_transformer_shard/~/layer_13/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_13/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_13/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_13/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_13/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_13/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_13/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_13/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_13/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_14/~/linear/w",
    "/params/causal_transformer_shard/~/layer_14/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_14/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_14/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_14/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_14/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_14/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_14/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_14/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_14/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_15/~/linear/w",
    "/params/causal_transformer_shard/~/layer_15/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_15/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_15/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_15/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_15/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_15/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_15/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_15/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_15/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_16/~/linear/w",
    "/params/causal_transformer_shard/~/layer_16/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_16/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_16/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_16/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_16/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_16/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_16/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_16/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_16/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_17/~/linear/w",
    "/params/causal_transformer_shard/~/layer_17/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_17/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_17/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_17/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_17/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_17/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_17/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_17/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_17/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_18/~/linear/w",
    "/params/causal_transformer_shard/~/layer_18/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_18/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_18/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_18/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_18/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_18/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_18/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_18/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_18/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_19/~/linear/w",
    "/params/causal_transformer_shard/~/layer_19/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_19/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_19/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_19/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_19/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_19/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_19/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_19/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_19/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_2/~/linear/w",
    "/params/causal_transformer_shard/~/layer_2/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_2/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_2/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_2/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_2/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_2/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_2/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_2/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_2/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_20/~/linear/w",
    "/params/causal_transformer_shard/~/layer_20/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_20/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_20/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_20/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_20/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_20/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_20/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_20/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_20/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_21/~/linear/w",
    "/params/causal_transformer_shard/~/layer_21/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_21/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_21/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_21/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_21/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_21/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_21/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_21/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_21/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_22/~/linear/w",
    "/params/causal_transformer_shard/~/layer_22/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_22/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_22/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_22/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_22/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_22/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_22/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_22/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_22/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_23/~/linear/w",
    "/params/causal_transformer_shard/~/layer_23/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_23/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_23/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_23/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_23/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_23/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_23/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_23/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_23/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_24/~/linear/w",
    "/params/causal_transformer_shard/~/layer_24/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_24/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_24/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_24/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_24/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_24/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_24/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_24/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_24/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_25/~/linear/w",
    "/params/causal_transformer_shard/~/layer_25/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_25/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_25/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_25/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_25/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_25/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_25/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_25/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_25/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_26/~/linear/w",
    "/params/causal_transformer_shard/~/layer_26/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_26/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_26/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_26/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_26/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_26/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_26/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_26/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_26/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_27/~/linear/w",
    "/params/causal_transformer_shard/~/layer_27/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_27/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_27/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_27/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_27/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_27/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_27/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_27/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_27/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_3/~/linear/w",
    "/params/causal_transformer_shard/~/layer_3/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_3/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_3/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_3/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_3/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_3/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_3/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_3/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_3/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_4/~/linear/w",
    "/params/causal_transformer_shard/~/layer_4/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_4/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_4/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_4/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_4/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_4/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_4/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_4/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_4/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_5/~/linear/w",
    "/params/causal_transformer_shard/~/layer_5/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_5/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_5/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_5/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_5/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_5/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_5/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_5/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_5/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_6/~/linear/w",
    "/params/causal_transformer_shard/~/layer_6/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_6/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_6/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_6/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_6/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_6/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_6/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_6/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_6/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_7/~/linear/w",
    "/params/causal_transformer_shard/~/layer_7/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_7/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_7/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_7/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_7/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_7/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_7/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_7/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_7/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_8/~/linear/w",
    "/params/causal_transformer_shard/~/layer_8/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_8/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_8/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_8/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_8/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_8/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_8/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_8/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_8/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/layer_9/~/linear/w",
    "/params/causal_transformer_shard/~/layer_9/~/linear_1/w",
    "/params/causal_transformer_shard/~/layer_9/~/linear_2/w",
    "/params/causal_transformer_shard/~/layer_9/~/linear_3/w",
    "/params/causal_transformer_shard/~/layer_9/~/linear_4/b",
    "/params/causal_transformer_shard/~/layer_9/~/linear_4/w",
    "/params/causal_transformer_shard/~/layer_9/~/linear_5/b",
    "/params/causal_transformer_shard/~/layer_9/~/linear_5/w",
    "/params/causal_transformer_shard/~/layer_9/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/layer_9/~/replicated_layer_norm/scale",
    "/params/causal_transformer_shard/~/projection_shard/~/linear/b",
    "/params/causal_transformer_shard/~/projection_shard/~/linear/w",
    "/params/causal_transformer_shard/~/projection_shard/~/replicated_layer_norm/offset",
    "/params/causal_transformer_shard/~/projection_shard/~/replicated_layer_norm/scale",
    "/step",
]


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


# TODO(dwarf): could be setup to load npz weights directly into hf model
# similar to `load_tf_weights_in_gpt2` in https://huggingface.co/transformers/v1.2.0/_modules/pytorch_transformers/modeling_gpt2.html
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

    # switch based on top level module name
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
    if len(x.shape) == 1:
        # out = x[0:1]
        out = np.array(x[0:1])

    elif len(x.shape) == 2:
        # print(f"LN/bias {x.shape}")

        # TODO(nijkamp): incorrect
        # if (x[1:] == x[-1]).all():
        if do_shard_ln or do_shard_bias:
            # print("LN")
            # if (x[1:] == 0).all() or (x[1:] == 1).all():
            if do_shard_ln:
                # TODO(nijkamp): for thise case, expression (x[1:] == 0).all() or (x[1:] == 1).all() should hold
                # out = x[0:1]
                out = np.array(x[0:1])
            else:
                # print("shard bias")
                # out = x[0:1] * x.shape[0] / old_shape[0]
                # TODO(nijkamp): sum() bias terms, is this correct?
                out = np.reshape(np.sum(x, axis=0), old_shape)
        else:
            # print("bias")
            out = x.reshape(old_shape)

    elif len(x.shape) == 3:
        # print(f"weight {x.shape}")
        if x.shape[0] * x.shape[2] == old_shape[2]:
            out = np.transpose(x, (1, 0, 2)).reshape(old_shape)
            # out = jnp.transpose(x, (1, 0, 2)).reshape(old_shape)
        elif x.shape[0] * x.shape[1] == old_shape[1]:
            # out = x.reshape(old_shape)
            out = np.reshape(x, old_shape)
        else:
            raise Exception(f"unimplemented, {x.shape}, {old_shape}")
    else:
        raise Exception(f"unimplemented, {x}")

    return out


def read_shard(ckpt_dir: FluidPath, pieces=16):
    out = []
    for idx in range(pieces):
        file_path = ckpt_dir / f"{idx}.npz"
        with file_path.open("rb") as f:
            buf = f.read()
            f_io = io.BytesIO(buf)
            deserialized = np.load(f_io)
            for i in deserialized:  # type: ignore
                out.append(deserialized[i])  # type: ignore
    return out


# def read_file_shards(ckpt_dir: FluidPath, fname: str, shards_in: int):
#     def read_npz(fpath: FluidPath):
#         with fpath.open("rb") as f:
#             buf = f.read()
#             f_io = io.BytesIO(buf)
#             return np.load(f_io)

#     # read same file accross shards
#     with multiprocessing.pool.ThreadPool(shards_in) as p:
#         return p.imap(read_npz, [ckpt_dir / f"shard_{i}" / fname for i in range(shards_in)])


# def lazy_read_ckpt_shards(ckpt_dir: FluidPath, shards_in: int, pieces=16):
#     for i in range(pieces):
#         fname = f"{i}.npz"
#         file_shards = read_file_shards(ckpt_dir, fname, shards_in)

#         # iterate over layers in file returning all shards for each
#         yield from zip(*file_shards)


def read_flattened_ckpt_with_names(
    old_flattened_pytree, input_ckpt: FluidPath, shards_in: int, shards_out: int
) -> Tuple[List[np.ndarray], List[str]]:
    global leaves_names_original
    global leaves_names_reduced

    # TODO(nijkamp): rewrite this mess
    with multiprocessing.pool.ThreadPool(shards_in) as p:
        print("Reading Shards (this could take a while)...")
        # load list of shards with axis/shape (n_shards(8?),n_layers,layer_shapes...)
        loaded_shards_in = list(p.imap(read_shard, [input_ckpt / f"shard_{i}" for i in range(shards_in)]))
        print("DONE reading shards")

    # transpose shards so that first index is layers and then shards
    # so that you can iterate through each layer and get all shards for that layer
    # new axis/shape (n_layers, n_shards(8?), layer_shapes...)
    loaded_shards_in = list(zip(*loaded_shards_in))

    #! continue work here. see if this is necessary and test on both cpu and gpu and tpu
    if len(loaded_shards_in) == len(leaves_names_original):
        matching_leave_names = leaves_names_original
    # reduced len=287
    elif len(loaded_shards_in) == len(leaves_names_reduced):
        matching_leave_names = leaves_names_reduced
    else:
        raise NotImplementedError(
            "Couldn't match loaded weights with corresponding leave names"
            f"{len(loaded_shards_in)=} {len(leaves_names_original)=} {len(leaves_names_reduced)=}"
        )

    unsharded_weights = []
    layer_names = []
    old_i = 0
    for i in tqdm(range(len(matching_leave_names)), desc="Resharding"):

        # pop instead of access to remove need to keep in memory
        leave_shards = loaded_shards_in.pop(0)
        leave_name = matching_leave_names[i]
        if leave_name.startswith("/opt_state"):
            continue

        old = old_flattened_pytree.pop(0)

        assert leave_name == leaves_names_reduced[old_i], f"{leave_name} {leaves_names_reduced[old_i]}"
        # old = old_flattened[old_i]
        old_i += 1

        x = np.stack(leave_shards)
        # TODO(nijkamp): what is this?
        if x.dtype == np.dtype("V2"):
            x.dtype = jnp.bfloat16

        if DEBUG:
            print(f"RESHARDING: {i=} {old_i=} {leave_name=} {x.shape=} {old.shape=}")

        if shards_out != shards_in:
            x = reshard(
                x,
                old.shape,
                do_shard_bias=leave_name.endswith("embedding_shard/~/linear/b")
                or leave_name.endswith("linear_5/b"),
                do_shard_ln=leave_name.endswith("replicated_layer_norm/offset")
                or leave_name.endswith("replicated_layer_norm/scale"),
            )

        unsharded_weights.append(x)
        layer_names.append(leave_name)

        assert x.shape == old.shape, f"Incompatible checkpoints {x.shape} vs {old.shape} {leave_name}"

    return unsharded_weights, layer_names


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


def save_hf_weights(
    pytree,
    input_ckpt: FluidPath,
    shards_in: int,
    shards_out: int,
    output_path: FluidPath,
    n_layers: int = 28,
):
    old_flattened, _ = jax.tree_flatten(pytree)
    del pytree
    unsharded, layer_names = read_flattened_ckpt_with_names(old_flattened, input_ckpt, shards_in, shards_out)

    # Convert to torch tensors at float16 precision.
    # Remove fist dimension which is 1 after resharding.
    # Transpose since all weights except wte require transposing for HF.
    unsharded = [torch.tensor(weights.squeeze(0).astype(np.float16)).half().T for weights in unsharded]

    wte_first = None

    pt_save_idx = 0
    save_map = {}
    for i in tqdm(range(len(unsharded)), desc="Saving pt files"):
        params = unsharded.pop(0)
        layer_name = layer_names.pop(0)

        hf_layer_id = leave_name_to_hf_layer_id(layer_name)
        if not hf_layer_id:
            continue

        # wte embedding weights need to be combined since hf model has no wte.embedding.bias
        if hf_layer_id.startswith("transformer.wte"):
            # un/re-transpose since wte weight is only layer that shouldn't be transposed
            params = params.T
            # store first weight/bias then skip saving
            if wte_first is None:
                wte_first = params
                continue
            # combine second wte bias/weight with first then move on to saving with weight name
            else:
                params = params + wte_first
                hf_layer_id = "transformer.wte.weight"

        pt_save_idx, save_map = save_hf_layer(params, hf_layer_id, pt_save_idx, output_path, save_map)

    # add attention bias layers
    # using float32 here instead of 16 to match pt model weights that were distributed for huggingface.
    attn_bias_weights = torch.tril(torch.tensor(np.ones((1, 1, 2048, 2048)), dtype=torch.float32))
    attn_masked_bias_weights = torch.tensor(np.array(-1e9), dtype=torch.float32)
    for i in range(n_layers):
        bias_id = f"transformer.h.{i}.attn.attention.bias"
        masked_bias_id = f"transformer.h.{i}.attn.attention.masked_bias"
        pt_save_idx, save_map = save_hf_layer(attn_bias_weights, bias_id, pt_save_idx, output_path, save_map)
        pt_save_idx, save_map = save_hf_layer(
            attn_masked_bias_weights, masked_bias_id, pt_save_idx, output_path, save_map
        )

    torch.save(save_map, (output_path / "m.pt").open(mode="wb"))


# expensive imports delayed until after command line argument validation
import jax
import jax.numpy as jnp

import optax
import mesh_transformer.util as util
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer


def save_sharded_to_hf_format(
    input_ckpt: Union[FluidPath, str],
    output_path: Union[FluidPath, str],
    cpu: bool = False,
):
    if cpu:
        jax.config.update("jax_platform_name", "cpu")

    input_ckpt, output_path = process_args(input_ckpt=input_ckpt, output_path=output_path)

    output_path.mkdir(exist_ok=True)

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
    }

    # TODO(nijkamp): overwriting the optimizer mutates the pytree in order to reduce memory alloc, but this will break the serialization format, serialize model into optim / param files separately to clean this mess
    params["optimizer"] = optax.scale(0)
    params["sampler"] = nucleaus_sample

    devices = np.array([jax.devices()[0]]).reshape((1, 1))
    with jax.experimental.maps.mesh(devices, ("dp", "mp")):
        network = CausalTransformer(params)

        save_hf_weights(
            network.state,
            input_ckpt=input_ckpt,
            shards_in=8,
            shards_out=1,
            output_path=output_path,
            n_layers=params["layers"],
        )


if __name__ == "__main__":
    # python to_hf_weights.py --input_ckpt ../gpt-j-train/base_models/step_383500 --output_path resharded/debug_ckpt --cpu
    save_sharded_to_hf_format(args["input_ckpt"], args["output_path"], args["cpu"])
