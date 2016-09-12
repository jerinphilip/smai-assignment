import numpy as np

sigmoid = lambda x: 1/(1+np.exp(-x))
sigmoid_f = (sigmoid, lambda x: np.multiply(sigmoid(x), (1-sigmoid(x))))


tanh = lambda x: 2/(1+np.exp(-2*x)) - 1
tanh_f = (tanh, lambda x: 1 - np.multiply(tanh(x), tanh(x)))
