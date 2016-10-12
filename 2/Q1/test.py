import du
import json
from P_utils import P

test_data = du.data('data/census-income.test')
headers = du.headers('data/column.keys')
classes = ['50000+.', '- 50000.']
stats = {}
with open("model.json", "r") as model:
    stats = json.load(model)

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
