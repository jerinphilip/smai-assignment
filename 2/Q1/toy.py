import du
import numpy as np
from functools import partial, reduce
from _util import plot_side_by_side, summarize, split_data
from P_utils import P
import json
from operator import or_, and_

def filter_f(data):
    return True
    bad = ['?' in data,
            #'Not in universe' in data
            ]
    return not(reduce(or_, bad))

headers = du.headers('data/column.keys')
data = du.data('data/census-income.data')
fdata = list(filter(filter_f, data))
test_data = du.data('data/census-income.test')
test_data = list(filter(filter_f, test_data))
classes = ['50000+.', '- 50000.']
train_data = fdata
#train_data, test_data = split_data(train_data, 0.1)
print(len(train_data), len(test_data))
stats = summarize(headers, train_data, classes)
"""with open('model.json', 'w') as model:
    outstring = json.dumps(stats, indent=4, sort_keys=True)
    print(outstring)
    model.write(outstring)
    #json.dump(stats, model, indent=4, sort_keys=True)
"""

keys = list(map(lambda x: x[0], headers))
total = len(test_data)
correct = 0
correct_class = {
        '- 50000.':0,
        '50000+.':0
        }
wrong_class = {
        '- 50000.':0,
        '50000+.':0
        }
for e in test_data:
    actual_cls = e[-1]
    t = list(zip(keys, e))
    t = t[:-1]
    maxcls, max_P = None, 0.0
    for cls in classes:
        P_cls = P(cls, t, stats)
        #print("Prob", cls, ":", P_cls)
        if P_cls >= max_P:
            maxcls, max_P = cls, P_cls
    if maxcls == actual_cls:
        correct += 1
        correct_class[actual_cls] += 1
    else:
        wrong_class[actual_cls] += 1

print("Accuracy:", (correct/total)*100, "%")
print("Hits:", correct, "Misses", total-correct)
print("Correct:", correct_class)
print("Wrong:", wrong_class)

