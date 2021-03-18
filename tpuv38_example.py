import os
import time
import jax
import numpy as np
import optax
import haiku as hk


from enwik8_loader import TextLoader
from mesh_transformer import checkpoint
from mesh_transformer.transformer_shard import CausalTransformer

bs = 8
seq = 1024
it = 1000

loader = TextLoader("data/enwik8", bs, seq)

devices = np.array(jax.devices()).reshape((1, 8))

import jax.profiler
server = jax.profiler.start_server(9999)
hk.experimental.profiler_name_scopes()

with jax.experimental.maps.mesh(devices, ('dp', 'mp')):
    opt = optax.chain(
        optax.clip_by_global_norm(1),
        optax.scale_by_adam(eps=1e-4),
        optax.scale(-1e-4),
    )

    start = time.time()
    
    # 2.7B
    c = CausalTransformer(dim=3072, heads=8, layer_count=12, vocab=256, optimizer=opt)
    
    # 4.8B
    # c = CausalTransformer(dim=4096, heads=32, layer_count=24, vocab=256, optimizer=opt)
    
    # 10B
    # c = CausalTransformer(dim=5120, heads=40, layer_count=32, vocab=256, optimizer=opt)

    # 8B-big-vocab
    # c = CausalTransformer(dim=5120, heads=40, layer_count=24, vocab=50400, optimizer=opt)

    param_count = hk.data_structures.tree_size(c.state['params'])

    print(f"Initialized in {time.time() - start:.06}s")
    print(f"Total parameters: {param_count}")

    start = time.time()
    for i in range(8):
        checkpoint.write_ckpt(c.state, f"gs://neo-models/mesh_jax/shard_{i}/", i)
    # checkpoint.write_ckpt((c.slow_state, c.fast_state), "gs://neo-models/mesh_jax/")
    # checkpoint.write_ckpt((c.slow_state, c.fast_state), "gs://neo-models/mesh_jax/")
    print(f"Checkpoint written in {time.time() - start:.06}s")

    start = time.time()
    sample = loader.get_samples()
    loss = c.train(sample)
    print(f"Compiled in {time.time() - start:.06}s")

    start = time.time()
    i = 0
    while True:
        with jax.profiler.StepTraceContext("train", step_num=i):
            sample = loader.get_samples()
            loss = c.train({
                "obs": sample[:, :-1],
                "target": sample[:, 1:],
            })
            if i % 100 == 0:
                print(f"it: {i}, loss: {loss.mean()}")

            if i % 100 == 0:
                start = time.time()
                for j in range(8):
                    checkpoint.write_ckpt(c.state, f"gs://neo-models/mesh_jax/shard_{j}/", j)
                print(f"Checkpoint written in {time.time() - start:.06}s")
        i += 1
