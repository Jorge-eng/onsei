import numpy as np
q = 15
M = 512
vt = 'int16_t'

def enforce_max_min(themax,themin,x):
    x[np.where(x > themax)] = themax
    x[np.where(x < themin)] = themin

maxval = (2**15) - 1
minval = -maxval

print maxval
print minval

k = np.arange(M+1) 

expw = np.exp(-1j * np.pi * k / M / 2.0)

A = 0.5 * (1.0 - expw)
B = 0.5 * (1.0 + expw)

Ar = np.round(np.real(A) * (2 ** q)).astype(int)
Ai = np.round(np.imag(A) * (2 ** q)).astype(int)

Br = np.round(np.real(B) * (2 ** q)).astype(int)
Bi = np.round(np.imag(B) * (2 ** q)).astype(int)

enforce_max_min(maxval,minval,Ar)
enforce_max_min(maxval,minval,Ai)
enforce_max_min(maxval,minval,Br)
enforce_max_min(maxval,minval,Bi)


Arstr = ','.join([str(x) for x in Ar])
Aistr = ','.join([str(x) for x in Ai])

Brstr = ','.join([str(x) for x in Br])
Bistr = ','.join([str(x) for x in Bi])

print 'const static %s Ar[%d] = {%s};\n' % (vt,len(A),Arstr)
print 'const static %s Ai[%d] = {%s};\n' % (vt,len(A),Aistr)

print 'const static %s Br[%d] = {%s};\n' % (vt,len(B),Brstr)
print 'const static %s Bi[%d] = {%s};\n' % (vt,len(B),Bistr)

