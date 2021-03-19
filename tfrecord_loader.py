import tensorflow as tf
import numpy as np


class TFRecordNewInputs:
    def __init__(self, index_fname, batch_size, sample_size, used=None, file_idx=0):
        if used is None:
            used = []

        self.file_idx = file_idx
        self.file_idx_init = False

        self.index = open(index_fname).read().splitlines()
        self.clean_index = list(filter(lambda x: x not in used, self.index))
        self.used = used
        self.bs = batch_size
        self.seq = sample_size

        self.sample_fn = self.sample_once()

    def sample_once(self):
        def tf_parse(example_proto):
            features = {
                "text": tf.io.VarLenFeature(tf.int64)
            }
            parsed_features = tf.io.parse_single_example(example_proto, features)
            return tf.sparse.to_dense(tf.sparse.reorder(parsed_features["text"])), parsed_features["text"].dense_shape[0]

        for i in self.clean_index:
            file = tf.data.TFRecordDataset(i).map(tf_parse).batch(np.prod(self.bs))
            for file_idx, (data, size) in enumerate(file):
                data = np.array(data)
                assert data.shape[-1] == self.seq + 1

                if not self.file_idx_init and file_idx <= self.file_idx:
                    continue
                self.file_idx_init = True
                self.file_idx = file_idx
                yield data.reshape(self.bs + (data.shape[-1],)).astype(np.uint32)
            self.used.append(i)
            self.file_idx = 0

    # this loops infinitely, use .sample_once to get an iterator for validation
    def get_samples(self):
        try:
            return next(self.sample_fn)
        except StopIteration:
            self.sample_fn = self.sample_once()
            return self.get_samples()


if __name__ == "__main__":
    d = TFRecordNewInputs("data/openwebtext2_new_inputs.val.index", (8, 32), 1024)
    for idx, i in enumerate(d.sample_once()):
        if idx % 1000 == 0:
            print(idx)

    print()
