import numpy as np
import matplotlib.pyplot as plt
#from gradient import gradient_descent_batch
#from gradient import J_batch

from gradient import *
from mse import closed

def process_data():
    data = open('data.txt', 'r')
    xs = []
    ts = []
    for line in data:
        data = list(map(float, line.strip().split(',')))
        x, t = data[:2], data[2]
        xs.append(x)
        ts.append(t)

    xs, ts = map(np.array, [xs, ts])
    return (xs, ts)



if __name__ == '__main__':
    xs, ts = process_data()
    n, d = xs.shape
    ys = np.c_[np.ones((n, 1)), xs]

    w1 = ys[ts > 0]
    w2 = ys[ts < 0]
    ys[ts < 0] = -1 * ys[ts < 0]
    # Visualize.
    fig, ax = plt.subplots()
    plt.scatter(w1[:,1], w1[:,2], color=['red'], marker='<', s=100)
    plt.scatter(w2[:,1], w2[:,2], color=['green'], s=100)
    jargin_args = [ys, 
                lambda x:2, 
                np.squeeze(np.random.rand(1, d+1)),
                1, J_single_relax()]
    #w = gradient_descent_single(*margin_args)
    #w = closed(ys, np.ones(n))
    print(np.squeeze(w.T))
    (c, a, b) = list(np.squeeze(w.T))
    xlims = ax.get_xlim()
    get_y = lambda x: (-a*x - c)/b
    ylims = list(map(get_y, xlims))
    plt.plot(xlims, ylims)
    plt.show()
