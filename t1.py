import tensorflow as tf
import numpy as np
from tfio import *

np.random.seed(1337)
x = np.arange(10, dtype=np.float32)[None].T.repeat(2, 1).reshape([-1, 2])
y = np.random.choice([0, 1, 2], size=10)
yr = np.random.normal(size=10)

nm = 'temp/a'
a1 = tfrec_name(nm, x.shape, 3)
a2 = tfrec_name(nm, x.shape, 0)
a3 = tfrec_pred_name(nm, x.shape)
# c = tfrec_img_name(nm, x.shape[0], 3, 10)
# d = tfrec_img_pred_name(nm, x.shape[0], 3, 1)

write_tfrec_from_array(x, y, nm, 3, num_examples_per_file=5)
write_tfrec_from_array(x, yr, nm, 0, num_examples_per_file=5)
write_tfrec_pred_from_array(x, nm, num_examples_per_file=5)

a1lst = rec_names(a1, 2)
a2lst = rec_names(a2, 2)
a3lst = rec_names(a3, 2)
gn1 = read_tfrec(a1lst, 4, None, True)
gn2 = read_tfrec(a2lst, 4, None, False)
gn3 = read_tfrec(a3lst, 4, None, False)


sess = tf.InteractiveSession()

