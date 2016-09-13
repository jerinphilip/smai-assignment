# Define
# layer = (W, f, dk)?
import numpy as np
from operator import eq, and_, or_
from functools import reduce
from activation import sigmoid_f, tanh_f



class layer:
    def __init__(self, x_d, y_d, **kwargs):
        #self.W = np.ones((y_d, x_d))
        self.W = np.random.randn(y_d, x_d)
        self.f, self.df = kwargs['activation']
        self.net = None
        self.x = None
        self.y = None
        self.sensitivity = None
        self.eta = kwargs['eta']

    def compute(self, x):
        self.x = x
        self.net = self.W.dot(x.T)
        self.y = self.f(self.net)
        return self.y
    
    def df_net(self):
        return self.df(self.net)

    def export_weights(self):
        return {"W": self.W}


class NeuralNet:
    def __init__(self, **kwargs):
        self.layers = []
        self.nlayers = 0
        self.z = None
        self.x = None

    def add_layer(self, x_d, y_d, **kwargs):
        l = layer(x_d, y_d, **kwargs)
        self.nlayers += 1
        self.layers.append(l)

    def _forward(self, x, n):
        l = self.layers[n]
        y = l.compute(x)
        if eq(n, self.nlayers-1):
            self.z = y
        else:
            self._forward(y, n+1)

    def forward(self, x):
        self.x = x
        self._forward(x, 0)

    def cross(self, d, y):
        d_ext = np.tile(d, (len(y), 1))
        y_ext = np.tile(y, (len(d), 1))
        matrix = np.multiply(d_ext.T, y_ext)
        return matrix


    def _backward(self, wTd_next, j):
        if j >= 0:
            l = self.layers[j]
            dj = np.multiply(wTd_next, l.df_net())
            dW = self.cross(dj, l.x)
            wTd = l.W.T.dot(dj.T)
            #wTd = l.W.T.dot(dk.T)
            l.W = l.W - l.eta * dW
            self._backward(wTd, j-1)

    def backward(self, t):
        op = self.layers[-1]
        dk = -(t-self.z)
        self._backward(dk, self.nlayers-1)


if __name__ == '__main__':
    N = NeuralNet()
    N.add_layer(2, 3, activation=sigmoid_f, eta=1e0)
    N.add_layer(3, 2, activation=sigmoid_f, eta=1e0)
    ip = np.array([1, 0])
    op = np.array([0.52,0.21])
    N.forward(ip)
    N.backward(op)
    for i in range(1000000):
        N.forward(ip)
        print(N.z)
        if(np.allclose(N.z, op)):
            break
        N.backward(op)
    print(N.layers[0].W)
    print(N.layers[1].W)
