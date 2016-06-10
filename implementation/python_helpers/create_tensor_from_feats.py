#!/usr/bin/python
import numpy as np
import tensor_utils
from matplotlib.pyplot import *

f = open(sys.argv[1],'r')
name = sys.argv[1].split(".")[0]

x = np.fromfile(f,dtype=np.int8)
f.close()
x = x.reshape(x.shape[0] / 40,40).astype(float)
x += 70.
x /= 140.
x = x.transpose()
x = x.reshape(1,1,x.shape[0],x.shape[1])

f = open(sys.argv[2],'w')
tensor_utils.write_fixed_point_tensor(name,x,f)
f.close()
