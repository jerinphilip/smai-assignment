# Define
# layer = (W, f, dk)?
import numpy as np
from operator import eq

class layer:
    def __init__(self, x_d, y_d, **kwargs):
        self.W = np.ones((y_d, x_d))
        self.f, self.df = kwargs['activation']
        self.net = None
        self.y = None
        self.sensitivity = None

    def compute(self, x):
        self.net = self.W.dot(x.T)
        self.y = self.f(self.net)
        return self.y
    
    def df_net(self):
        return self.df(self.net)


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

if __name__ == '__main__':
    N = NeuralNet()
    sigmoid = lambda x: 1/(1+np.exp(x))
    ds = lambda x: sigmoid(x)*(1-sigmoid(x))
    N.add_layer(2, 2, activation=(sigmoid, ds))
    N.add_layer(2, 2, activation=(sigmoid, ds))
    x = np.array([1,2])
    y = np.array([1,1])
    N.forward(x)
    print(N.z)
    print(N.x)
