import tensorflow as tf
import numpy as np
from tfio import *

np.random.seed(1337)
hs = np.random.choice([1, 2], size=10)
ws = np.random.choice([1, 2], size=10)

x_lst = []
for i in range(10):
    # x = np.ones(shape=[hs[i], ws[i], 2], dtype=np.float32) * i
    x = np.random.normal(size=[hs[i], ws[i], 2]).astype(np.float32)
    x_lst.append(x)

y = np.random.choice([0, 1, 2], size=10)
yr = np.random.normal(size=10)

dgn1 = zip(x_lst, y)
dgn2 = zip(x_lst, yr)
dgn3 = iter(x_lst)

nm = 'temp/d'

a1 = write_tfimgrec_from_generator(dgn1, nm, 3, num_examples_per_file=None)
a2 = write_tfimgrec_from_generator(dgn2, nm, 0, num_examples_per_file=6)
a3 = write_tfimgrec_pred_from_generator(dgn3, nm, num_examples_per_file=5)


#gn0 = read_raw_tfimgrec(a1, None)
gn1 = read_tfimgrec(a1, (2, 2), 4, None, shuffle=False)
gn2 = read_tfimgrec(a2, (2, 2), 4, None, shuffle=False)
gn3 = read_tfimgrec(a3, (2, 2), 4, None, shuffle=False)

sess = tf.InteractiveSession()

a1a = convert_tfimgrec_to_tfrec(sess, a1, (2, 2))
a2a = convert_tfimgrec_to_tfrec(sess, a2, (2, 2))
a3a = convert_tfimgrec_to_tfrec(sess, a3, (2, 2))

gn1a = read_tfrec(a1a, 4, None, False)
gn2a = read_tfrec(a2a, 4, None, False)
gn3a = read_tfrec(a3a, 4, None, False)









