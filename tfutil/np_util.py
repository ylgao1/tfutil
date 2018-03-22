import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

__all__ = ['weighted_spearman', 'weighted_pearson', 'one_hot', 'split_test_data',
           'split_test_data_r', 'split_val_and_test_data', 'split_val_and_test_data_r',
           'generate_data', 'generate_data_for_testing']

def weighted_pearson(x, y, w=None):
    if w is None:
        w = np.ones_like(x)
    y_demean = y - np.average(y, weights=w)
    x_demean = x - np.average(x, weights=w)
    corr = ((np.sum(w * y_demean * x_demean) / np.sum(w)) /
            np.sqrt((np.sum(w * y_demean ** 2) * np.sum(w * x_demean ** 2)) / (np.sum(w) ** 2)))
    return corr


def weighted_spearman(x, y, w=None, method='average'):
    if w is None:
        w = np.ones_like(x)
    yr = rankdata(y, method)
    xr = rankdata(x, method)
    return weighted_pearson(xr, yr, w)


def one_hot(indices, depth):
    m = np.zeros([len(indices), depth])
    m[np.arange(len(indices)), indices] = 1
    return m


def split_test_data(y, test_ratio):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio)
    tri = None
    tei = None
    for itr, ite in sss.split(np.zeros(len(y)), y):
        tri = itr
        tei = ite
    return tri, tei


def split_val_and_test_data(y, val_ratio, test_ratio):
    '''
    Data will be splitted into 3 pieces: train, val and test
    val_ratio: #val / (#train + #val)
    test_ratio: #test / #all
    '''
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio)
    tri_raw = None
    tri = None
    tvi = None
    tei = None
    for itr, ite in sss_test.split(np.zeros(len(y)), y):
        tri_raw = itr
        tei = ite
    ytr = y[tri_raw]
    for itr, itv in sss_val.split(np.zeros(len(ytr)), ytr):
        tri = tri_raw[itr]
        tvi = tri_raw[itv]
    return tri, tvi, tei


def split_test_data_r(n, test_ratio):
    ss = ShuffleSplit(n_splits=1, test_size=test_ratio)
    tri = None
    tei = None
    for itr, ite in ss.split(np.zeros(n), np.arange(n)):
        tri = itr
        tei = ite
    return tri, tei


def split_val_and_test_data_r(n, val_ratio, test_ratio):
    ss_test = ShuffleSplit(n_splits=1, test_size=test_ratio)
    ss_val = ShuffleSplit(n_splits=1, test_size=val_ratio)
    tri_raw = None
    tri = None
    tvi = None
    tei = None
    y = np.arange(n)
    for itr, ite in ss_test.split(np.zeros(n), y):
        tri_raw = itr
        tei = ite
    ytr = y[tri_raw]
    for itr, itv in ss_val.split(np.zeros(len(ytr)), ytr):
        tri = tri_raw[itr]
        tvi = tri_raw[itv]
    return tri, tvi, tei


def generate_data(x, y, batch=32, shuffle=True, allow_smaller_final_batch=False):
    num_examples = len(y)
    idxes = np.arange(num_examples)
    N = num_examples // batch
    res = num_examples % batch
    while True:
        if shuffle:
            np.random.shuffle(idxes)
        for i in range(N):
            ib = idxes[i * batch: (i + 1) * batch]
            yield (x[ib], y[ib])
        if res != 0:
            if allow_smaller_final_batch:
                ib = idxes[-res:]
                yield (x[ib], y[ib])


def generate_data_for_testing(x, y, batch=32):
    num_examples = len(y)
    idxes = np.arange(num_examples)
    N = num_examples // batch
    res = num_examples % batch
    for i in range(N):
        ib = idxes[i * batch: (i + 1) * batch]
        yield (x[ib], y[ib])
    if res != 0:
        ib = idxes[-res:]
        yield (x[ib], y[ib])
