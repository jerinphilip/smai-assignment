from operator import eq, and_, or_, mod
import numpy as np
from matplotlib import pyplot as plot

def single_sample(xs, ts, eta, margin=0):
    # Take dim
    n, d = xs.shape
    # Pad with 1
    ys = np.c_[np.ones((n, 1)), xs]

    # The points, same space
    ys[ts < 0 ] = -1*ys[ ts < 0 ]
    w = np.random.randn(1, d+1)
    pass_number = 0

    while True:
        pass_number = pass_number + 1
        updates = 0
        for y in ys:
            h_x = np.squeeze(y.dot(w.T))
            if h_x <= margin:
                updates = updates + 1
                w = w + y
        print("Pass %d Updates %d"%(pass_number, updates))
        if eq(updates, 0): break

    return w

def batch_perceptron(xs, ts, eta, margin=0):
    n, d = xs.shape
    ys = np.c_[np.ones((n, 1)), xs]
    ys[ts < 0] = -1 * ys[ ts < 0 ]

    w = np.random.randn(1, d+1)
    k = 0
    while True:
        k = k + 1
        h_x = np.squeeze(ys.dot(w.T))
        misclassified = list(ys[ h_x <= margin])
        # Update weights.
        w = w + sum(misclassified, 0)
        if (not misclassified):
            break
    print("Finished in %d iterations"%(k))
    return w


