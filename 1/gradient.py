import numpy as np
from operator import eq

class J_batch:
    def derivative(self, ys, a, b):
        h_x = np.squeeze(ys.dot(a.T))
        misclassified = list(ys[ h_x <= b ])
        return -np.sum(misclassified, 0)

class J_single:
    def derivative(self, y, a, b):
        return -y

    def value(self, y, a, b):
        return np.squeeze(y.dot(a.T))

class J_single_relax:
    def derivative(self, y, a, b):
        A = b - np.squeeze(y.dot(a.T))
        B = np.linalg.norm(y)
        C = (A)/(B*B)
        h_x  = C * y
        return -1*h_x

    def value(self, y, a, b):
        A = y.dot(a.T) - b
        N = np.linalg.norm(y)
        return 0.5*(A*A)/(N*N)


def gradient_descent_batch(ys, eta, a, b, J, threshold=0):
    k = 0
    while True:
        k = k+1
        da = eta(k) * J.derivative(ys, a, b)
        a = a  - da
        if  np.linalg.norm(da) <= threshold:
            break
    return a

def gradient_descent_single(ys, eta, a, b, J, threshold=0.000001):
    k = 0
    while True:
        k = k+1
        misclassified = 0
        prev_a = a
        for y in ys:
            if y.dot(a.T) <= b:
                misclassified = misclassified + 1
                da = eta(k) * J.derivative(y, a, b)
                a = a - da
                if (np.linalg.norm(da) <= threshold):
                    print("Misclassified=", misclassified)
                    return a

        if eq(misclassified, 0):
            break

    return a

def widrow_hoff(ys, eta, a, b, J, threshold=0.00000001):
    eta = lambda x: 2e0
    k = 0
    while True:
        k = k+1
        misclassified = 0
        prev_a = a
        for y in ys:
            if y.dot(a.T) <= b: misclassified = misclassified + 1
            da = eta(k) * (b - y.dot(a.T))
            print(b - y.dot(a.T))
            a = a - da * y
            print(da, y)
            #print(np.linalg.norm(da) ,threshold)
            #print((np.linalg.norm(da) <threshold))
            if (np.linalg.norm(da) <= threshold):
                print("Misclassified=", misclassified)
                return a

        if eq(misclassified, 0):
            break

    return a
