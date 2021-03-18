import ray

from typing import Callable
import numpy as np

from mesh_transformer.train_actor import NetworkRunner


class TPUCluster:
    def __init__(self,
                 mesh_shape,
                 node_count,
                 model: Callable):
        assert ray.is_initialized()  # needs a valid ray cluster to start
        self.nodes = []
        self.node_count = node_count
        for i in range(node_count):
            self.nodes.append(NetworkRunner.options(max_concurrency=2).remote(mesh_shape, model))

        for n in self.nodes:
            n.run.remote()

    def train(self, data):
        data_chunks = np.array_split(data, len(self.nodes), axis=1)

        res = []
        for n, d in zip(self.nodes, data_chunks):
            res.append(n.train.remote({
                "obs": d[:, :, :-1],
                "target": d[:, :, 1:],
            }))

        return np.array(ray.get(res)).mean()
