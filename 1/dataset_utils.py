import cv2
import numpy as np
from scipy.misc import imresize
from matplotlib import pyplot as plt
from neuralnet import NeuralNet
from activation import sigmoid_f, tanh_f

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
    #truth = np.zeros(10).astype(np.float_)
    #truth[value] = 1
    return (img, value)

    vectorized = list(filter(lambda x: x[1] in digits, vectorized))
def argmx(l):
    mx = 0
    for i in range(len(l)):
        if l[i] > l[mx]:
            mx = i
    return mx

