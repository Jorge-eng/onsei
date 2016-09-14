import numpy as np


'''
General idea is to compute interpolation table for tanh

so on interval of say.... [0, xend), subdivide into size 2^N intervals
at each interval, compute tanh at x_i and x_i+1, 
compute slope = [ y_i+1 - y_i ] / [ x_i+1 - x_i] 
and y-intercept y_i 

'''
x_end = 4.0
N = 10
Q = 15

dx = np.array(x_end * (2 ** Q) / (2 ** N)).astype(int)
xvals = (np.arange(1 + 2**N) * dx).astype(int)

slopes = []
yvals = []
for i in range(1,len(xvals)):
    x2 = xvals[i]
    x1 = xvals[i - 1]
    xx1 = float(x1) / (2**Q)
    xx2 = float(x2) / (2**Q)

    y1 = np.tanh(xx1)
    y2 = np.tanh(xx2)

    slope = (y2 - y1) / (xx2 - xx1)

    yvals.append(int(y1* (2**Q)))
    slopes.append(int(slope* (2**Q)))

slopestr = ','.join([str(d) for d in slopes])
yvalstr = ','.join([str(d) for d in yvals])
print 'const static int16_t k_tanh_slopes[%d] = {%s};\n' % (2**N,slopestr)
print 'const static int16_t k_tanh_yvals[%d] = {%s};\n' % (2**N,yvalstr)
print 'there are %d intervals'  % 2**N
