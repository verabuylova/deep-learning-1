from sklearn.utils import shuffle
import numpy as np
import math

class DataLoader(object):
    """
    Tool for shuffling data and forming mini-batches
    """
    def __init__(self, X, y, batch_size=1, shuffle=False):
        """
        :param X: dataset features
        :param y: dataset targets
        :param batch_size: size of mini-batch to form
        :param shuffle: whether to shuffle dataset
        """
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_id = 0

    def __len__(self) -> int:
        """
        :return: number of batches per epoch
        """
        return int(np.ceil(len(self.X) / self.batch_size))

    def num_samples(self) -> int:
        """
        :return: number of data samples
        """
        return len(self.X)

    def __iter__(self):
        """
        Shuffle data samples if required
        :return: self
        """
        if self.shuffle:
            self.X, self.y = shuffle(self.X, self.y, random_state=0)
            self.batch_id = 0
        return self


    def __next__(self):
        """
        Form and return next data batch
        :return: (x_batch, y_batch)
        """
            
        start = self.batch_id * self.batch_size
        end = min(start + self.batch_size, len(self.X))
        x_batch, y_batch = self.X[start:end], self.y[start:end]

        self.batch_id += 1
        if start >= len(self.X):
            self.batch_id = 0
            raise StopIteration

        return x_batch, y_batch

        