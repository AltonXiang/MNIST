import numpy as np
from tqdm import tqdm
import module.nn as nn


class SGD(object):
    def __init__(self, net: nn.Module, lr=0.001, weight_decay=None):
        self.lr = lr
        self.weight_decay = weight_decay
        self.net = net

    def step(self):
        for layer in self.net.layers:
            if layer.requires_grad:
                grad = layer.weight_grad.copy()
                if self.weight_decay is not None:
                    grad += layer.weight * self.weight_decay
                layer.weight -= self.lr * grad

                grad = layer.bias_grad.copy().reshape(1, -1)
                if self.weight_decay is not None:
                    grad += layer.bias * self.weight_decay
                layer.bias -= self.lr * grad
