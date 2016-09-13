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

def onehot(x, m):
    y = np.zeros(m).astype(np.float_)
    y[x] = 1.0
    return y


def train_net(filename, maxIter, nH, digits):
    digits_file = open(filename, 'r')
    processed = list(read_data(digits_file))
    downsampled = map(downsample, processed)
    vectorized = list(map(vectorize, downsampled))
    filtered = list(filter(lambda x: x[1] in digits, vectorized))
    #Input output dimensions
    m = len(filtered[0][0])
    p = len(digits)

    N = NeuralNet()

    N.add_layer(m, nH, activation=sigmoid_f, eta=2e0)
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
            opv = onehot(digits.index(op), p)
            N.backward(opv)
            error += np.sum(np.square((opv - N.z)))
        print("Error:", error)

    return N

def validate_net(N, filename, digits):
    digits_file = open(filename, 'r')
    processed = list(read_data(digits_file))
    downsampled = map(downsample, processed)
    vectorized = list(map(vectorize, downsampled))
    filtered = list(filter(lambda x: x[1] in digits, vectorized))
    negatives = 0
    for (ip, op) in filtered:
        N.forward(ip)
        a, b = digits[argmx(N.z)], op
        if (a !=b): negatives += 1
    return (negatives, len(filtered))




if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    # The answer to life, universe and everything.
    np.random.seed(42)
    digits = [3, 4, 5]
    N = train_net(args.train_set, 100, 10, digits)
    negatives, total = validate_net(N, args.validation_set, digits)
    print("Negatives: %d/%d"%(negatives, total))
    np.save("weights", N.export_net())


