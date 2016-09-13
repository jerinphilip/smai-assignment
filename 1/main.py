import argparse
from dataset_utils import read_data, downsample, vectorize, argmx
from neuralnet import NeuralNet
from activation import sigmoid_f, tanh_f
import numpy as np

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

def train_net(filename, maxIter):
    digits = open(filename, 'r')
    processed = list(read_data(digits))
    downsampled = map(downsample, processed)
    vectorized = list(map(vectorize, downsampled))
    N = NeuralNet()
    N.add_layer(65, 20, activation=sigmoid_f, eta=1e0)
    #N.add_layer(20, 20, activation=sigmoid_f, eta=1e0)
    N.add_layer(20, 10, activation=sigmoid_f, eta=1e0)
    total = len(vectorized)
    for i in range(maxIter):
        negatives = 0
        #if (i+1)%100 == 0: print("Iter", i+1)

        error = 0.0
        counter = -1
        for (ip, op) in vectorized:
            counter += 1
            N.forward(ip)
            N.backward(op)
            error += np.sum(np.square((op - N.z)))
        print("Error:", error)

    return N

def validate_net(N, filename):
    digits = open(filename, 'r')
    processed = list(read_data(digits))
    downsampled = map(downsample, processed)
    vectorized = list(map(vectorize, downsampled))
    negatives = 0
    for (ip, op) in vectorized:
        N.forward(ip)
        a,b = argmx(N.z), argmx(op)
        if (a !=b):
            negatives += 1
            #print(a, b)
    return (negatives, len(vectorized))




if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    # Return Neural Net learning from Input
    N = train_net(args.train_set, 500)
    negatives, total = validate_net(N, args.validation_set)
    print("Negatives: %d/%d"%(negatives, total))
