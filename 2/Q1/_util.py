from functools import partial
from P_utils import gaussian, bin_column
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from random import randint

def plot_side_by_side(stats):
    for stat in stats:
        mu = stat["mean"]
        sigma = stat["std"]
        k = 2 
        x = np.linspace(mu-k*sigma, mu+k*sigma, 10000)
        y = norm.pdf(x, mu, sigma)
        plt.plot(x, y)
    plt.show()


def summarize(headers, data, classes):
    stat_op = {
        "continuous": gaussian,
        "nominal": bin_column
    }
    data_d = {}
    ofClass = lambda cls, entry: cls == entry[-1]

    for cls in classes:
        filter_f = partial(ofClass, cls)
        data_d[cls] = list(filter(filter_f, data))


    htitles, htypes = zip(*headers)

    stats = { cls:{} for cls in classes }

    for cls in classes:
        columnwise = zip(*data_d[cls])
        obj = zip(htitles,htypes,columnwise)

        for (htitle, htype, column) in obj:
            stats_d = {"type": htype}
            stats_d = stat_op[htype](column, stats_d)
            stats[cls][htitle] = stats_d
    return stats


def split_data(ls, fraction):
    xs = [[], []]
    max_count = [int(fraction*len(ls)), len(ls)-int(fraction*len(ls))]
    count = [0, 0]
    for e in ls:
        i = randint(0, 1)
        if (count[i] >= max_count[i]):
            i = int(not(i))
        count[i] += 1
        xs[i].append(e)
    return xs



