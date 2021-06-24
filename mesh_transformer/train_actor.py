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
        print(f"jax runtime initialization starting")
        import jax
        from jax.experimental.maps import thread_resources, ResourceEnv, Mesh
        import haiku as hk
        from mesh_transformer.checkpoint import write_ckpt, read_ckpt
        # jax.experimental.maps.EXPERIMENTAL_SPMD_LOWERING = True

        thread_resources.env = ResourceEnv(Mesh(np.empty((), dtype=object), ()))

        start = time.time()
        # print(jax.devices())
        print(f"jax devices: {jax.device_count()}")
        print(f"jax runtime initialized in {time.time() - start:.06}s")
        devices = np.array(jax.devices()).reshape(self.mesh_shape)

        with jax.experimental.maps.mesh(devices, ('dp', 'mp')):
            start = time.time()
            network = self.network_builder()
            param_count = hk.data_structures.tree_size(network.state['params'])
            print(f"Initialized in {time.time() - start:.06}s")
            print(f"Total parameters: {param_count}")

            while True:
                operation, input = self.input_q.get()
                if operation == "train":
                    self.output_q.put(network.train(input))
                elif operation == "eval":
                    self.output_q.put(network.eval(input))
                elif operation == "generate":
                    self.output_q.put(network.generate(*input))
                elif operation == "write_ckpt":
                    path, shard = input
                    write_ckpt(network.state, path, shard)
                    self.output_q.put(None)
                elif operation == "load_ckpt":
                    network.state = read_ckpt(network.state, input, devices.shape[1])
                    self.output_q.put(network.state["step"][0])
                elif operation == "get_params":
                    self.output_q.put(hk.data_structures.tree_size(network.state['params']))
                elif operation == "move_params":
                    # only needed for inference, otherwise first train step does this
                    local_shards = max(jax.local_device_count() // self.mesh_shape[1], 1)

                    # delete the optimizer states otherwise it OOMs for some reason
                    # TODO: use ShardedDeviceArray or something to get around this for bigger models
                    del network.state["opt_state"]
                    network.state = network.move_xmap(network.state, np.zeros(local_shards))
                    self.output_q.put(None)
                else:
                    raise Exception("Not implemented")

    def get_params(self):
        self.input_q.put(("get_params", None))
        return self.output_q.get()

    def train(self, sample):
        self.input_q.put(("train", sample))
        return self.output_q.get()

    def eval(self, sample):
        self.input_q.put(("eval", sample))
        return self.output_q.get()

    def generate(self, input):
        self.input_q.put(("generate", input))
        return self.output_q.get()

    def write_ckpt(self, path, shard):
        self.input_q.put(("write_ckpt", (path, shard)))
        return self.output_q.get()

    def load_ckpt(self, path):
        self.input_q.put(("load_ckpt", path))
        return self.output_q.get()

    def move_params(self):
        self.input_q.put(("move_params", None))
        return self.output_q.get()
