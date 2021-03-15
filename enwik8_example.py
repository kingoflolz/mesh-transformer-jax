import os

# Run with a bunch of CPU devices.
def setUpModule():
    global prev_xla_flags
    prev_xla_flags = os.getenv("XLA_FLAGS")
    flags_str = prev_xla_flags or ""
    if "xla_force_host_platform_device_count" not in flags_str:
        os.environ["XLA_FLAGS"] = (flags_str + " --xla_force_host_platform_device_count=8")


setUpModule()

import jax
import numpy as np
import optax
import haiku as hk
import util

from transformer_shard import CausalTransformer

from loader import TextLoader

loader = TextLoader("data/enwik8", 512, 64)

devices = np.array(jax.devices()).reshape((2, 4))

with jax.experimental.maps.mesh(devices, ('dp', 'mp')):
    opt = optax.chain(
        optax.clip_by_global_norm(1),
        optax.scale_by_adam(eps=1e-4),
        optax.scale(-1),
        optax.scale_by_schedule(util.gpt3_schedule(1_000, 20_000, 1e-4, 1e-5))
    )

    c = CausalTransformer(dim=512, heads=8, layer_count=8, vocab=256, optimizer=opt)

    print(f"Total parameters: {hk.data_structures.tree_size(c.state['params'])}")

    for i in range(2000):
        sample = loader.get_samples()
        loss = c.train(sample)

        print(f"it: {i}, loss: {loss.mean()}")

    jax.profiler.save_device_memory_profile("memory.pprof")
