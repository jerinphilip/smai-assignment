import cv2
import numpy as np
import scipy as sp
from scipy.misc import imresize
from matplotlib import pyplot as plt

def process_chunk(chunk):
    imgstr = chunk[:32]
    value = int(chunk[32])
    convert = lambda x: list(map(lambda y: int(y), x))
    img = np.array(list(map(convert, imgstr)))
    return (img, value)


def read_data(input_file):
    lines = list(map(lambda x: x.strip(), input_file.readlines()))
    description = lines[:21]
    data = lines[21:]
    chunks = [data[i:i+33] for i in range(0, len(data), 33)]
    processed = map(process_chunk, chunks)
    return processed

def downsample(packed):
    img, value = packed
    img = imresize(img, 0.25, 'nearest')
    img = np.floor(img/255).astype(np.integer)
    return (img, value)

if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    digits = open(filename, 'r')
    processed = read_data(digits)
    downsampled = map(downsample, processed)
    
