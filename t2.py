import tensorflow as tf
import numpy as np
from tfio import *

np.random.seed(1337)
hs = np.random.choice([1, 2], size=10)
ws = np.random.choice([1, 2], size=10)

x_lst = []
for i in range(10):
    x = np.ones(shape=[hs[i], ws[i], 2], dtype=np.float32) * i
    x_lst.append(x)

y = np.random.choice([0, 1, 2], size=10)
yr = np.random.normal(size=10)

nm = 'temp/b'
a1 = tfimgrec_name(nm, len(x_lst), 2, 3)
a2 = tfimgrec_name(nm, len(x_lst), 2, 0)
a3 = tfimgrec_pred_name(nm, len(x_lst), 2)

write_tfimgrec_from_lst(x_lst, y, nm, 3, num_examples_per_file=5)
write_tfimgrec_from_lst(x_lst, yr, nm, 0, num_examples_per_file=5)
write_tfimgrec_pred_from_lst(x_lst, nm, num_examples_per_file=5)