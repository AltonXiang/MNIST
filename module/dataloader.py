import numpy as np


class DataLoader(object):
    def __init__(self, data, batch_size=64, shuffle=True):
        self.images, self.labels = np.array(data[0]).reshape(len(data[0]), -1)/255, np.array(data[1]).reshape(len(data[1]), -1)
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        if isinstance(index, np.ndarray):
            return self.images[index], self.labels[index]
        if isinstance(index, slice):
            return self.images[index], self.labels[index]
        if isinstance(index, int):
            if index < 0:
                index += len(self)
            if index < 0 or index >= len(self):
                raise IndexError("The index (%d) is out of range." % index)
            return self.images[index], self.labels[index]

    def __iter__(self):
        indices = np.arange(len(self))
        if self.shuffle:
            np.random.shuffle(indices)
        self.images, self.labels = self.images[indices], self.labels[indices]
        self.n = 0
        return self

    def __next__(self):
        if self.n >= len(self):
            raise StopIteration
        batch = self.images[self.n:min(len(self), self.n+self.batch_size)], self.labels[self.n:min(len(self), self.n+self.batch_size)]
        self.n += self.batch_size
        return batch
