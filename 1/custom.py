from neuralnet import NeuralNet
from functools import reduce
from operator import eq, and_
import numpy as np

if __name__ == '__main__':
    from itertools import product
    N = NeuralNet()
    sigmoid = lambda x: 1/(1+np.exp(-x))
    ds = lambda x: np.multiply(sigmoid(x), (1-sigmoid(x)))
    
    f_id = lambda x: x
    df_id = lambda x: 1
    tpl = (sigmoid, ds)


    N.add_layer(2, 2, activation=tpl, eta=1e1)
    #N.add_layer(3, 3, activation=(sigmoid, ds), eta=1e0)
    N.add_layer(2, 1, activation=tpl, eta=1e1)

    inputs = list(product([0,1],[0,1]))
    print(inputs)
    outputs = list(map(lambda x:reduce(and_, x), inputs))
    pairs = list(zip(inputs, outputs))

    for (ip, op) in pairs:
        print(ip, "-", op)

    for i in range(1000000):
        if((i+1)%1000 == 0 or i<100):
            print("Iteration", i+1)
        for (ip, op) in pairs:
            ip = np.array(ip)
            op = np.array([op])
            N.forward(ip)
            N.backward(op)
            #s = input()
            if((i+1)%1000 == 0 or i<100):
                print(N.x, "->", N.z)

