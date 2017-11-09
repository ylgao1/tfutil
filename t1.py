import tensorflow as tf
import numpy as np
from tfio import *

x = np.random.normal(size=[10, 4, 5])
y1 = np.random.choice([0, 1, 2], size=10)
y2 = np.random.normal(size=10)

nm = 'temp/data/abc'

a = tfrec_name(nm, x.shape, 0, 1)
b = tfrec_pred_name(nm, x.shape, 1)
c = tfrec_img_name(nm, x.shape[0], 3, 10, 1)
d = tfrec_img_pred_name(nm, x.shape[0], 3, 1)

# write_tfrec_from_array(x, y1, 'temp/a1', 3)
# write_tfrec_from_array(x, y1, 'temp/a2', 0)
