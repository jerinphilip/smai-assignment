# Define
# layer = (W, f, dk)?
import numpy as np
from operator import eq, and_
from functools import reduce


class layer:
    def __init__(self, x_d, y_d, **kwargs):
        self.W = np.ones((y_d, x_d))
        #self.W = np.random.randn(y_d, x_d)
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

    def export(self):
        pass


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


    def _backward(self, dk, j):
        if j >= 0:
            l = self.layers[j]

            dj = np.multiply(dk, l.df_net())
            dW = self.cross(dj, l.x)

            wTd = l.W.T.dot(dj.T)

            l.W = l.W - l.eta * dW
            self._backward(wTd, j-1)

    def backward(self, t):
        op = self.layers[-1]
        dk = np.multiply(-(t-self.z), op.df_net())
        dW = self.cross(dk, op.x)
        wTd = op.W.T.dot(dk.T)
        op.W = op.W - op.eta * dW
        self._backward(wTd, self.nlayers-2)

if __name__ == '__main__':
    N = NeuralNet()
    sigmoid = lambda x: 1/(1+np.exp(-x))
    ds = lambda x: np.multiply(sigmoid(x),(1-sigmoid(x)))
    
    f_id = lambda x: x
    df_id = lambda x: 1
    tpl = (sigmoid, ds)

    N.add_layer(2, 3, activation=tpl, eta=1e0)
    #N.add_layer(2, 2, activation=(sigmoid, ds), eta=1e1)
    N.add_layer(3, 2, activation=tpl, eta=1e0)
    ip = np.array([1, 2])
    op = np.array([0.02, 0.93])
    N.forward(ip)
    N.backward(op)
    for i in range(1000000):
        N.forward(ip)
        print(N.z)
        N.backward(op)
