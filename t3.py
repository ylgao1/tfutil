import numpy as np
from tfio import *


np.random.seed(1337)
x = np.arange(10, dtype=np.float32)[None].T.repeat(2, 1).reshape([-1, 2])
y = np.random.choice([0, 1, 2], size=10)
yr = np.random.normal(size=10)

