import ray

@ray.remote(resources={"tpu": 1})
class NetworkRunner(object):
    def __init__(self, mesh_shape, network_builder):
        import jax

        with jax.experimental.maps.mesh(devices, ('dp', 'mp')):
            jax.devices()
            print("jax runtime initialized")

            start = time.time()
            c = network_builder()
            param_count = hk.data_structures.tree_size(c.state['params'])
            print(f"Initialized in {time.time() - start:.06}s")
            print(f"Total parameters: {param_count}")

    start = time.time()
    sample = loader.sample()
    loss = c.train(sample)
    print(f"Compiled in {time.time() - start:.06}s")

    start = time.time()
    for i in range(it):
        with jax.profiler.StepTraceContext("train", step_num=i):
            sample = loader.sample()
            loss = c.train(sample)
            if i % 10 == 0:
                print(f"it: {i}, loss: {loss.mean()}")
    total_time = time.time() - start
    print(f"{it} steps in {total_time:.06}s")

    weight_flops = bs * seq * it * param_count
    attn_flops = bs * (seq ** 2) * it * 32 * 5120 * 16
    print(f"effective flops (not including attn): {weight_flops * 6 / total_time:.06}")
    print(f"MXU flops: {(weight_flops * 8 + attn_flops) / total_time:.06}")
    jax.profiler.save_device_memory_profile("memory.pprof")
