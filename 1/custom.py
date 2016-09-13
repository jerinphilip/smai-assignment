from neuralnet import NeuralNet
from functools import reduce
from itertools import product
from operator import eq, and_, or_
import numpy as np
from activation import sigmoid_f, tanh_f

if __name__ == '__main__':
    from itertools import product
    N = NeuralNet()
    np.random.seed(1)
    #N.add_layer(2, 2, activation=sigmoid_f, eta=1e2)
    N.add_layer(3, 3, activation=sigmoid_f, eta=1e0)
    N.add_layer(3, 3, activation=sigmoid_f, eta=1e0)
    N.add_layer(3, 3, activation=sigmoid_f, eta=1e2)
    N.add_layer(3, 1, activation=sigmoid_f, eta=1e0)

    inputs = list(product([0, 1], [0, 1]))
    outputs = list(map(lambda x: and_(x[0], x[1]), inputs))
    inputs = list(map(lambda x: [1]+list(x), inputs))
    pairs = list(zip(inputs, outputs))
    print(pairs)

    for (ip, op) in pairs:
        print(ip, "-", op)

    for i in range(1000000):
        if((i+1)%1000 == 0 or i<100): print("Iteration", i+1)
        for (ip, op) in pairs:
            ip = np.array(ip)
            op = np.array(op)
            N.forward(ip)
            N.backward(op)
            if((i+1)%1000 == 0 or i<100): print(N.x, "->", N.z)

