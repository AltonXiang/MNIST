import module
import module.function as F
import module.dataloader
from module.data import load_data_mnist
import module.nn as nn
import module.optim as optim
from module.train import train
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.linalg

class MLP(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, activation='relu'):
        super(MLP, self).__init__()
        if activation == 'relu':
            self.layers = [F.Linear(num_inputs, num_hiddens), F.ReLU(), F.Linear(num_hiddens, num_outputs), F.Softmax()]
        elif activation == 'sigmoid':
            self.layers = [F.Linear(num_inputs, num_hiddens), F.Sigmoid(), F.Linear(num_hiddens, num_outputs), F.Softmax()]
        else:
            raise NotImplementedError

with open('./model/mnist_model.pkl','rb') as f:
    net = pickle.load(f)

#可视化该神经网络每一层的网络参数，清晰地看到每一层的网络参数的分布情况
def visualize(net):
    for i, layer in enumerate(net.layers):
        if layer.requires_grad:
            weight = layer.weight
            bias = layer.bias
            if i == 0:
                plt.figure(figsize=(10, 10))
                _extracted_from_visualize_8(1, 'weight of layer 1', weight)
                _extracted_from_visualize_8(2, 'bias of layer 1', bias)
            elif i == 2:
                _extracted_from_visualize_8(3, 'weight of layer 2', weight)
                _extracted_from_visualize_8(4, 'bias of layer 2', bias)
    plt.show()

def _extracted_from_visualize_8(arg0, arg1, arg2):
    plt.subplot(2, 2, arg0)
    plt.title(arg1)
    plt.hist(arg2.flatten(), bins=100)


class Visual():
    def __init__(self):
        self.arr = None
        self.w = None
    
    def fit(self, X):
        Sigma = np.cov(X, rowvar=True)
        ret = scipy.linalg.schur(Sigma, sort=lambda x: abs(x))
        self.w, self.arr = ret[0], ret[1]
        self.w = np.diag(self.w)
    
    def transform(self, X, n):
        return np.dot(self.arr[:, 1:min(len(self.w), n+1)].T, X)
    
    def fit_transform(self, X, n):
        self.fit(X)
        return self.transform(X, n)

visual = Visual()

def visual_weight1(net):
    weight_fc1 = visual.fit_transform(net.layers[0].weight.transpose(), 36).reshape(36, 28, 28)
    for i in range(36):
        plt.subplot(6, 6, i+1)
        plt.imshow(weight_fc1[i])
        plt.axis('off')
        plt.gray()
    plt.show()

def visual_weight2(net):
    weight_fc2 = net.layers[2].weight.transpose().reshape(10, -1)
    for i in range(10):
        plt.subplot(10, 1, i+1)
        plt.imshow(np.tile(weight_fc2[i], (10, 1)))
        plt.axis('off')
        plt.gray()
    plt.show()

#可视化神经网络第一层的bias
def visual_bias1(net):
    bias_fc1 = net.layers[0].bias.reshape(16,16)
    for i in range(16):
        plt.subplot(16, 1, i+1)
        plt.imshow(np.tile(bias_fc1[i], (16, 1)))
        plt.axis('off')
        plt.gray()
    plt.show()

#可视化神经网络第二层的bias
def visual_bias2(net):
    bias_fc2 = net.layers[2].bias.transpose().reshape(10, -1)
    for i in range(10):
        plt.subplot(10, 1, i+1)
        plt.imshow(np.tile(bias_fc2[i], (10, 1)))
        plt.axis('off')
        plt.gray()
    plt.show()



visualize(net)
visual_weight1(net)
visual_weight2(net)
visual_bias1(net)
visual_bias2(net)




