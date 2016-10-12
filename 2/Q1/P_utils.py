import numpy as np
from math import exp, sqrt, pi
from functools import partial

def bin_column(column, stats):
    bad = ['Not in universe', '?']
    #bad = ['?']
    total = 0
    for entry in column:
        if entry not in bad:
            total += 1
            if entry not in stats:
                stats[entry] = 0
            stats[entry] += 1

    # Normalize
    for entry in stats:
        if(entry not in ['type', '50000+.', '- 50000.']):
            stats[entry] = stats[entry]/total
    return stats


def gaussian(column, stats):
    fvalues = list(map(float, column))
    A = np.array(fvalues)
    result = {
        "mean": np.mean(A),
        "var": np.var(A),
        "std": np.std(A)
    }
    stats.update(result)
    return stats

def P_gauss(d, x):
    mu = d["mean"]
    sigma = d["std"]
    return exp(-(x-mu)**2/(2*(sigma**2)))/(sqrt(2*pi)*sigma)

def Pi(model, condition):
    X, value = condition
    if ( model[X]['type'] == 'continuous'):
        return 1.0
        value = float(value)
        return P_gauss(model[X], value)
    if value in ['Not in universe', '?']:
        return 1.0
    if value not in model[X]:
        return 0.0;
    return model[X][value]

def P(label, conditions, model):
    p_xC = partial(Pi, model[label])
    entries = [model[cls]['class'][cls] for cls in model.keys()]
    prior = model[label]['class'][label]/sum(entries)
    Ps = np.array(list(map(p_xC, conditions)))
    #P_q = np.array([prior]*len(Ps))
    Ps = np.append([prior], Ps)
    #ppS = np.prod(Ps)/np.prod(P_q)
    #return prior
    ppS = np.prod(Ps)
    return ppS
    logPs = np.log(1+Ps)
    #logPs = np.log(Ps)
    slogPs = np.sum(logPs)
    return slogPs

