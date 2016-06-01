#!/usr/bin/python
import numpy as np
N = 400

h = np.hanning(N)
x = np.round(h * 32767)

print ','.join(x.astype(int).astype(str).tolist())
