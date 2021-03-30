import time

import haiku as hk
import numpy as np
import optax

from enwik8_loader import TextLoader
from mesh_transformer.sampling import softmax_sample
from mesh_transformer.transformer_shard import CausalTransformer

bs = 8
seq = 1024
it = 10000

# batch = (gas, batch)
loader = TextLoader("data/enwik8", (1, bs), seq)

import jax.profiler

server = jax.profiler.start_server(9999)
hk.experimental.profiler_name_scopes()

opt = optax.chain(
    optax.clip_by_global_norm(1),
    optax.scale_by_adam(eps=1e-4),
    optax.scale(-1e-4),
)

start = time.time()

# model that compiles quickly for testing
tiny = {
    "layers": 1,
    "d_model": 512,
    "n_heads": 4,
    "cores_per_replica": 4
}

# 25M
base = {
    "layers": 8,
    "d_model": 512,
    "n_heads": 4,
    "cores_per_replica": 4
}

# 200M
large = {
    "layers": 16,
    "d_model": 1024,
    "n_heads": 8,
    "cores_per_replica": 8
}

# 2.7B
gpt_2b7 = {
    "layers": 16,
    "d_model": 3072,
    "n_heads": 12,
    "cores_per_replica": 8
}

# 4.8B
gpt_4b8 = {
    "layers": 24,
    "d_model": 4096,
    "n_heads": 32,
    "cores_per_replica": 8
}

# 10B
gpt_10b = {
    "layers": 32,
    "d_model": 5120,
    "n_heads": 40,
    "cores_per_replica": 8
}

config = base
config["n_vocab"] = 256
config["norm"] = "layernorm"
config["seq"] = 1024
config["optimizer"] = opt
config["sampler"] = softmax_sample

devices = np.array(jax.devices()).reshape((-1, config["cores_per_replica"]))

with jax.experimental.maps.mesh(devices, ('dp', 'mp')):
    c = CausalTransformer(config)

    param_count = hk.data_structures.tree_size(c.state['params'])

    print(f"Initialized in {time.time() - start:.06}s")
    print(f"Total parameters: {param_count}")

    start = time.time()
    sample = loader.get_samples()
    loss = c.train({
                "obs": sample[:, :, :-1],
                "target": sample[:, :, 1:],
            })
    print(f"Compiled in {time.time() - start:.06}s")

    start = time.time()
    for i in range(it):
        with jax.profiler.StepTraceContext("train", step_num=i):
            sample = loader.get_samples()
            loss = c.train({
                "obs": sample[:, :, :-1],
                "target": sample[:, :, 1:],
            })

            def list_to_str(x):
                return ''.join(chr(char) for char in x)

            if i % 100 == 0:
                print(f"it: {i}, loss: {np.array(loss).mean()}")
                input_sample = loader.get_samples()
                _, (output, _) = c.generate(input_sample[0, :, :-1], np.ones(bs) * seq, 100)
                output = output[:, :, 0]
                for sample in output[:1]:
                    print(f"ctx: {repr(list_to_str(input_sample[0, 0, -100:].tolist()))}")
                    print(f"gen: {repr(list_to_str(sample.tolist()))}")

    total_time = time.time() - start
    print(f"{it} steps in {total_time:.06}s")

    weight_flops = bs * seq * it * param_count
    attn_flops = bs * (seq ** 2) * it * 32 * 5120 * 16
    print(f"effective flops (not including attn): {weight_flops * 6 / total_time:.06}")
    print(f"MXU flops: {(weight_flops * 8 + attn_flops) / total_time:.06}")
    jax.profiler.save_device_memory_profile("memory.pprof")
