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

    

def draw_graph(plt, ax, w, title):
    (c, a, b) = list(np.squeeze(w.T))
    xlims = ax.get_xlim()
    get_y = lambda x: (-a*x - c)/b
    ylims = list(map(get_y, xlims))
    line, = plt.plot(xlims, ylims, label=title)
    return line



if __name__ == '__main__':
    xs, ts = process_data()
    n, d = xs.shape
    ys = np.c_[np.ones((n, 1)), xs]
    #np.random.seed(42)

    w1 = ys[ts > 0]
    w2 = ys[ts < 0]
    ys[ts < 0] = -1 * ys[ts < 0]
    # Visualize.
    fig, ax = plt.subplots()
    plt.scatter(w1[:,1], w1[:,2], color=['red'], marker='<', s=100)
    plt.scatter(w2[:,1], w2[:,2], color=['green'], s=100)
    relax_margin_args = [ys, 
                lambda x:2.0, 
                np.squeeze(np.random.rand(1, d+1)),
                1, J_single_relax(), 0.0001]
    no_margin_args = [ys, lambda x:1.0, np.squeeze(np.random.rand(1, d+1)), 1.0, J_single(), 0]
    margin_args = [ys, lambda x:1.0, np.squeeze(np.random.rand(1, d+1)), 10, J_single(), 0]

    handles = []
    print("Without margin")
    w = gradient_descent_single(*no_margin_args)
    l = draw_graph(plt, ax, np.squeeze(w.T), "Single sample without margin")
    handles.append(l)
    print("With margin")
    w = gradient_descent_single(*margin_args)
    l = draw_graph(plt, ax, np.squeeze(w.T), "Single sample with margin")
    handles.append(l)
    print("With margin relax")
    w = gradient_descent_single(*relax_margin_args)
    l = draw_graph(plt, ax, np.squeeze(w.T), "Single sample with margin relaxation")
    handles.append(l)

    print("Widrow Hoff")
    w = widrow_hoff(*relax_margin_args)
    l = draw_graph(plt, ax, np.squeeze(w.T), "Widrow Hoff Rule")
    handles.append(l)
    #print("LMS PseudoInverse")
    #w = closed(ys, np.random.randn(n))
    #l = draw_graph(plt, ax, np.squeeze(w.T), "LMS, pseudoinverse")
    #handles.append(l)
    plt.legend(handles=handles)
    plt.show()
