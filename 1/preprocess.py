import cv2
import numpy as np
import scipy as sp
from scipy.misc import imresize
from matplotlib import pyplot as plt
from neuralnet import NeuralNet
from activation import sigmoid_f

def process_chunk(chunk):
    """ process_chunk: String->(np.2darray, int) """
    imgstr = chunk[:32]
    value = int(chunk[32])
    convert = lambda x: list(map(lambda y: int(y), x))
    img = np.array(list(map(convert, imgstr)))
    return (img, value)


def read_data(input_file):
    """ read_data: File -> [(np.2darray, int)] """
    lines = list(map(lambda x: x.strip(), input_file.readlines()))
    description = lines[:21]
    data = lines[21:]
    chunks = [data[i:i+33] for i in range(0, len(data), 33)]
    processed = map(process_chunk, chunks)
    return processed

def downsample(packed):
    """ downsample: [(np.2darray, int)] -> [(np.2darray, int)] """
    img, value = packed
    img = imresize(img, 0.25, 'nearest')
    img = np.floor(img/255).astype(np.integer)
    return (img, value)

def vectorize(packed):
    """ vectorize: [(np.2darray, int)] -> [(np.1darray, np.1darray)]"""
    img, value = packed
    img = np.append([1.0], img.ravel().astype(np.float_))
    truth = np.zeros(10).astype(np.float_)
    truth[value] = 1
    return (img, truth)

def argmx(l):
    mx = 0
    for i in range(len(l)):
        if l[i] > l[mx]:
            mx = i
    return mx

if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    digits = open(filename, 'r')
    processed = list(read_data(digits))
    downsampled = map(downsample, processed)
    vectorized = list(map(vectorize, downsampled))
    N = NeuralNet()
    N.add_layer(65, 20, activation=sigmoid_f, eta=1e0)
    #N.add_layer(32, 20, activation=tpl, eta=1e1)
    N.add_layer(20, 10, activation=sigmoid_f, eta=1e0)
    total = len(vectorized)
    #vectorized = vectorized[:1]
    for i in range(100000):
        negatives = 0
        if (i+1)%100 == 0:
            print("Iter", i+1)

        counter = -1
        for (ip, op) in vectorized:
            counter += 1
            N.forward(ip)
            N.backward(op)
            if (i+1)%100 == 0 or i<100:
                a,b = argmx(op), argmx(N.z)
                if a!=b: 
                    print(a, "->", b)
                    #plt.imshow(processed[counter][0], 'gray')
                    #plt.imshow(ip.reshape(8, 8), cmap='gray')
                    #plt.text(0, 0, "Orig %d, Id %d"%(a, b))
                    #plt.show()
                    negatives += 1

        if (i+1)%100 == 0 or i<100:
            print("Negatives:", negatives, "/", total)

