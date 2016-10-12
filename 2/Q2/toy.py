import numpy as np
from numpy.linalg import eig
from functools import partial
from gnb import stats
from math import pi, sqrt, exp

def sparse_to_vec(size, M):
    result = np.zeros(size).astype(np.float64)
    for i in M:
        result[i-1] = 1e0
    return result

def convert(row):
    row = row.strip()
    row = list(map(int, row.split(' ')))
    return row

def aSort(r):
    l,e = r
    idx = l.argsort()[::-1]
    return (l[idx], e[:,idx])

data_file = 'data/dorothea_train.data'
def read_data(data_file):
    data_fp = open(data_file, 'r')
    data = list(map(convert, data_fp))
    f = partial(sparse_to_vec, 10**5)
    data = np.array(list(map(f, data)))
    return data

data = read_data(data_file).astype(np.float64)

def transform(data_matrix, eigenbasis):
    return data_matrix.dot(eigenbasis)

def take(v, k):
    return v[:, :k]

dS = (data - np.mean(data, axis=0))
S_f = dS.dot(dS.T)
l, e = aSort(eig(S_f))
v = dS.T.dot(e)
v = take(v, 50)
data_c = transform(data, v)
np.save('data', data_c)

label_file = 'data/dorothea_train.labels'
def read_labels(label_file):
    label_fp = open(label_file, 'r')
    labels = np.array(list(map(int, label_fp)))
    return labels
labels = read_labels(label_file)

classes = [1, -1]
data_d = {}
stats_d = {}
count_d = {}

for cls in classes:
    idx = labels == cls
    data_d[cls] = data_c[idx, :]
    count_d[cls] = len(data_d[cls])

for cls in classes:
    stats_d[cls] = stats(data_d[cls])

#print(stats_d)

def probability(entry, stat):
    def Pi(X):
        x, m, s, v = X
        return exp(-(x-m)**2/(2*v))/(sqrt(2*pi)*s)

    mean, std, var = stat
    sub_entry = zip(entry, mean, std, var)
    Ps = np.array(list(map(Pi, sub_entry)))
    return np.prod(Ps)

def argmax(d):
    mx, mv = None, -1
    for x in d:
        if d[x] > mv:
            mx, mv = x, d[x]
    return mx

test_file = 'data/dorothea_valid.data'
test_data = read_data(test_file)
test_data_c = transform(test_data, v)
test_label_file = 'data/dorothea_valid.labels'
test_labels = read_labels(test_label_file)

correct, total = 0, 0
correct_s = {1:0, -1:0}
wrong_s = {1: 0, -1:0}
#for entry in zip(test_data_c, test_labels):
for entry in zip(data_c, labels):
    test_input, label = entry
    P = {}
    for cls in classes:
        prior = count_d[cls]/sum([count_d[x] for x in count_d])
        P[cls] = prior*probability(test_input, stats_d[cls])
    #print(argmax(P), label)
    total += 1
    if(argmax(P) == label):
        correct += 1
        correct_s[label] +=1
    else:
        wrong_s[label] += 1

print(correct, "/", total)
print("Accuracy: ", 100*correct/total)
print("Correct", correct_s)
print("Wrong", wrong_s)
