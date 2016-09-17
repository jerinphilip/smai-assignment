import argparse
from dataset_utils import read_data, downsample, vectorize, argmx
from neuralnet import NeuralNet
from activation import sigmoid_f, tanh_f
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import matplotlib.colors
from functools import partial, reduce
import random

def create_parser():
    arguments = [
        ['-i', '--input', "input filename", str, "train_set"],
        ['-v', '--validate', "validation set", str, 'validation_set']
    ]

    parser = argparse.ArgumentParser(description="Helper Script")
    for arg in arguments:
        unix, gnu, desc, typename, dest = arg
        parser.add_argument(unix, gnu, help=desc, type=typename, 
                required=True, dest=dest)
    return parser

def onehot(x, m):
    y = np.zeros(m).astype(np.float_)
    y[x] = 1.0
    return y

def binary(x, l):
    bstr = bin(x)[2:]
    length = len(bstr)
    padding = (l-length)*'0'
    binform = list(map(int, padding+bstr))
    return np.array(binform).astype(np.float_)

def bin2dec(x):
    y = np.rint(x).astype(np.integer).tolist()
    bstr = ''.join(list(map(str, y)))
    return int(bstr, 2)


def train_net(filename, maxIter, nH, digits):
    digits_file = open(filename, 'r')
    processed = list(read_data(digits_file))
    downsampled = map(downsample, processed)
    vectorized = list(map(vectorize, downsampled))
    filtered = list(filter(lambda x: x[1] in digits, vectorized))
    #Input output dimensions
    m = len(filtered[0][0])
    p = len(bin(len(digits)))-2

    N = NeuralNet()

    N.add_layer(m, nH, activation=sigmoid_f, eta=2e0)
    #N.add_layer(nH, nH, activation=sigmoid_f, eta=2e0)
    N.add_layer(nH, p, activation=sigmoid_f, eta=2e0)
    total = len(filtered)
    for i in range(maxIter):
        negatives = 0
        #if (i+1)%100 == 0: print("Iter", i+1)

        error = 0.0
        counter = -1
        for (ip, op) in filtered:
            counter += 1
            N.forward(ip)
            #opv = onehot(digits.index(op), p)
            opv = binary(digits.index(op), p)
            #print(digits.index(op), opv)
            N.backward(opv)
            error += np.sum(np.square((opv - N.z)))
        print("Iteration %d, Error:"%(i), error)

    return N

def validate_net(N, filename, digits):
    digits_file = open(filename, 'r')
    processed = list(read_data(digits_file))
    downsampled = map(downsample, processed)
    vectorized = list(map(vectorize, downsampled))
    filtered = list(filter(lambda x: x[1] in digits, vectorized))
    negatives = 0
    export = []

    for (ip, op) in filtered:
        N.forward(ip)
        #a, b = digits[argmx(N.z)], op
        
        if bin2dec(N.z) < len(digits):
            a, b = digits[bin2dec(N.z)], op
        else:
            a, b = -1, op
        export.append((a, b, N.layers[1].x))
        if (a !=b): negatives += 1

    return (negatives, len(filtered), export)

def visualize(exported):
    obtained, expected, ys = list(zip(*exported))
    length = len(obtained)
    pca = PCA(n_components=2)
    ys_t = pca.fit_transform(ys)
    ys_table = [[] for i in range(length)]
    for i in range(length):
        ys_table[expected[i]].append(ys_t[i,:])

    colors = list(matplotlib.colors.cnames.values()) 
    random.shuffle(colors)
        
    for i in range(length):
        if ys_table[i]:
            ys_i = np.array(ys_table[i])
            plt.scatter(ys_i[:,0], ys_i[:, 1], c=colors[i], marker='x', s=100)

    centers = [np.ones(2) for i in range(length)]
    for i in range(length):
        if(ys_table[i]):
            centers[i] = np.sum(ys_table[i], axis=0)/len(ys_table[i])
    centers = np.array(centers)

    n_digits = len(set(expected))
    kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
    kmeans.fit(ys_t)
    #ys_kpt = K.fit_predict(np.array(ys_t))dd
    centroids = kmeans.cluster_centers_
    centroids = centers
    for i in range(length):
        if ys_table[i]:
            x, y = centroids[i]
            plt.scatter(x, y, marker='o', s=400, linewidths=10, color=colors[i], zorder=10)
            plt.annotate(
		str(i), xy = (x, y), xytext = (-20, 20),
		textcoords = 'offset points', ha = 'right', va = 'bottom',
		bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
		arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

    plt.show()
    


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    # The answer to life, universe and everything.
    np.random.seed(42)
    #digits = [3, 4, 5]
    digits = list(range(10))
    N = train_net(args.train_set, 60, 64, digits)
    negatives, total, exported = validate_net(N, args.validation_set, digits)
    print("Negatives: %d/%d"%(negatives, total))
    np.save("weights", N.export_net())
    visualize(exported)
    #cluster(exported)


