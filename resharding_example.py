# This was tested with an RTX 3090, peak memory usage is approximately 22.4GB during inference, and 19GB when loading the model
# The following environment variables were also used: XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform

import time

import jax
from jax.experimental import maps
import numpy as np
import optax
import transformers

from mesh_transformer.checkpoint import read_ckpt
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer

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
  "cores_per_replica": 1,  # only running on one GPU
  "per_replica_batch": 1,
}

per_replica_batch = params["per_replica_batch"]
cores_per_replica = params["cores_per_replica"]
seq = params["seq"]


params["sampler"] = nucleaus_sample

# here we "remove" the optimizer parameters from the model (as we don't need them for inference)
params["optimizer"] = optax.scale(0)

devices = np.array([jax.devices()[0]]).reshape((1, 1))
maps.thread_resources.env = maps.ResourceEnv(maps.Mesh(devices, ('dp', 'mp')))

tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')

network = CausalTransformer(params)

start = time.time()

# here we load a checkpoint which was written with 8 shards into 1 shard
network.state = read_ckpt(network.state, "step_383500/", 8, shards_out=cores_per_replica)

# move the state to CPU/system memory so it's not duplicated by xmap
network.state = jax.device_put(network.state, jax.devices("cpu")[0])

def infer(context, top_k=40, top_p=0.9, temp=1.0, gen_len=512):
    tokens = tokenizer.encode(context)

    provided_ctx = len(tokens)
    pad_amount = seq - provided_ctx

    padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)
    batched_tokens = np.array([padded_tokens] * per_replica_batch)
    length = np.ones(per_replica_batch, dtype=np.uint32) * len(tokens)

    start = time.time()
    output = network.generate(batched_tokens, length, gen_len, {"top_p": np.ones(per_replica_batch) * top_p, "top_k": top_k is not None and (np.ones(per_replica_batch, dtype=np.int32) * top_k) or None, "temp": np.ones(per_replica_batch) * temp})

    samples = []
    decoded_tokens = output[1][0]

    for o in decoded_tokens[:, :, 0]:
      samples.append(tokenizer.decode(o))

    print(f"completion done in {time.time() - start:06}s")
    return samples


infer("EleutherAI is")
