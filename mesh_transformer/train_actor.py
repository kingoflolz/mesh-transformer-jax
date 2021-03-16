import threading

import ray
import time
import numpy as np
from queue import Queue


@ray.remote(resources={"tpu": 1})
class NetworkRunner(object):
    def __init__(self, mesh_shape, network_builder):
        self.mesh_shape = mesh_shape
        self.network_builder = network_builder

        self.input_q = Queue(maxsize=1)
        self.output_q = Queue(maxsize=1)

    def run(self):
        import jax
        from jax.experimental.maps import thread_resources, ResourceEnv, Mesh
        import haiku as hk

        thread_resources.env = ResourceEnv(Mesh(np.empty((), dtype=object), ()))

        start = time.time()
        print(jax.devices())
        print(jax.device_count())
        print(f"jax runtime initialized in {time.time() - start:.06}s")
        devices = np.array(jax.local_devices()).reshape(self.mesh_shape)
        print(devices)

        with jax.experimental.maps.mesh(devices, ('dp', 'mp')):
            start = time.time()
            network = self.network_builder()
            param_count = hk.data_structures.tree_size(network.fast_state['params'])
            print(f"Initialized in {time.time() - start:.06}s")
            print(f"Total parameters: {param_count}")

            while True:
                operation, input = self.input_q.get()
                if operation == "train":
                    self.output_q.put(network.train(input))
                elif operation == "update":
                    network.update(input)
                    self.output_q.put(None)

    def train(self, sample):
        self.input_q.put(("train", sample))
        return self.output_q.get()

    def update(self, div):
        self.input_q.put(("update", div))
        self.output_q.get()
