import pickle as pk
import numpy as np

# TODO: add validation set
class Data:

    def __init__(self, filename=None):
        self._samples_used = 0
        self._epochs_completed = 0
        if filename: self.load(filename)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.train_x, self.train_t, self.test_x, self.test_t = pk.load(f)

    def next_batch(self, size):
        if self._samples_used >= self.num_samples:
            self._epochs_completed += 1
            self.reset()
        start = self._samples_used
        end = min(start + size, len(self.train_x))
        self._samples_used = end
        return self.train_x[start:end], self.train_t[start:end]

    def reset(self):
        perm = np.arange(self.num_samples)
        np.random.shuffle(perm)
        self.train_x = self.train_x[perm]
        self.train_t = self.train_t[perm]
        self._samples_used = 0

    @property
    def num_samples(self):
        return self.train_x.shape[0]

    @property
    def sample_size(self):
        return self.train_x.shape[1]

    @property
    def samples_shape(self):
        return self.train_x.shape

    @property
    def num_targets(self):
        return self.train_t.shape[0]

    @property
    def target_size(self):
        return self.train_t.shape[1]

    @property
    def targets_shape(self):
        return self.train_t.shape

    @property
    def epochs_completed(self):
        return self._epochs_completed
