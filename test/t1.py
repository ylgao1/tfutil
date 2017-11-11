import tensorflow as tf
import numpy as np
from tfio import *

np.random.seed(1337)
x = np.arange(10, dtype=np.float32)[None].T.repeat(2, 1).reshape([-1, 2])
y = np.random.choice([0, 1, 2], size=10)
yr = np.random.normal(size=10)

nm = 'temp/a'


a1 = write_tfrec_from_array(x, y, nm, 3, num_examples_per_file=5)
a2 = write_tfrec_from_array(x, yr, nm, 0, num_examples_per_file=5)
a3 = write_tfrec_pred_from_array(x, nm, num_examples_per_file=5)


gn1 = read_tfrec(a1, 4, None, True)
gn2 = read_tfrec(a2, 4, None, False)
gn3 = read_tfrec(a3, 4, None, False)


sess = tf.InteractiveSession()

