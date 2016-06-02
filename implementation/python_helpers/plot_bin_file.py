#!/usr/bin/python
import sys
from matplotlib.pyplot import *
import numpy as np


f = open(sys.argv[1],'r')
x = np.fromfile(f,dtype=np.int8)
print np.max(x),np.min(x)
x = x.reshape(x.shape[0] / 40,40)
X,Y = np.meshgrid(range(x.shape[1]),range(x.shape[0]))
pcolormesh(Y.transpose(),X.transpose(),x.transpose()); xlabel('time (10ms frames)'); ylabel('mel bins'); show()
f.close()

