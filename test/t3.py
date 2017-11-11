import numpy as np
from tfio import *


np.random.seed(1337)
x = np.arange(10, dtype=np.float32)[None].T.repeat(4, 1).reshape([-1, 2, 2])
y = np.random.choice([0, 1, 2], size=10)
yr = np.random.normal(size=10)

dgn1 = zip(x, y)
dgn2 = zip(x, yr)
dgn3 = iter(x)

nm = 'temp/c'

a1 = write_tfrec_from_generator(dgn1, nm, num_classes=3, num_examples_per_file=6)
a2 = write_tfrec_from_generator(dgn2, nm, num_classes=0, num_examples_per_file=6)
a3 = write_tfrec_pred_from_generator(dgn3, nm, num_examples_per_file=5)

gn1 = read_tfrec(a1, 4, None, True)
gn2 = read_tfrec(a2, 4, None, False)
gn3 = read_tfrec(a3, 4, None, False)

sess = tf.InteractiveSession()