from mnist import MNIST
import time
import numpy as np
from module.dataloader import DataLoader

def load_data_mnist(batch_size=64, shuffle=True):
    mndata = MNIST('./data')
    train_iter = DataLoader(mndata.load_training(), batch_size, shuffle)
    test_iter = DataLoader(mndata.load_testing(), batch_size)
    train_len, test_len = len(train_iter), len(test_iter)
    img_size, num_classes = train_iter.images.shape[1], 10
    print(f'Loading MNIST dataset. {train_len} items for training, {test_len} items for testing. batch_size={batch_size}')
    print(f'X size: {img_size}, MNIST image scale 28x28, total_cls={num_classes}.')
    return train_iter, test_iter, img_size, num_classes

if __name__ == "__main__":
    train_iter, test_iter, _, _ = load_data_mnist()
    for iterator, name in ((train_iter, 'train_iter'), (test_iter, 'test_iter')):
        t0 = time.perf_counter()
        for _ in iterator:
            continue
        t1 = time.perf_counter()
        print(f'{t1 - t0} for {name}')
