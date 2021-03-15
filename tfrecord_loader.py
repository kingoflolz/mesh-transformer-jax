import tensorflow as tf
import numpy as np


class TFRecordTokenizerDataset:
    def __init__(self, index, used, bs, seq):
        self.index = index
        self.clean_index = list(filter(lambda x: x not in used, index))
        self.used = used
        self.bs = bs
        self.seq = seq

    def sample(self):
        def tf_parse(example_proto):
            features = {
                "text": tf.io.VarLenFeature(tf.int64)
            }
            parsed_features = tf.io.parse_single_example(example_proto, features)
            return tf.sparse.to_dense(tf.sparse.reorder(parsed_features["text"])), parsed_features["text"].dense_shape[0]

        for i in self.clean_index:
            file = tf.data.TFRecordDataset(i).map(tf_parse).batch(self.bs)
            for data, size in file:
                data = np.array(data)

                yield {"obs": data[:, :-1].astype(np.uint32), "target": data[:, 1:].astype(np.uint32)}
            self.used.append(i)

    def infinite(self):
        while True:
            for i in self.sample():
                yield i
            self.used = []
            self.clean_index = self.index[:]


if __name__ == "__main__":
    index = open("data/openwebtext2_new_inputs.train.index").read().splitlines()

    d = TFRecordTokenizerDataset(index, [], 32, 1024)
    for idx, i in enumerate(d.sample()):
        if idx % 1000 == 0:
            print(idx)

    print()
