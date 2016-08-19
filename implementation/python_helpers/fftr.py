import numpy as np

def fftr(x,M):
    xc = x[::2] + 1j * x[1::2]

    z = np.fft.fft(xc)
    z = np.concatenate((z,[z[0]]))

    k = np.arange(M+1)

    expw = np.exp(-1j * np.pi * k / M / 2.0)

    A = 0.5 * (1.0 - expw)
    B = 0.5 * (1.0 + expw)

    G = A * z + B * np.conj(z[::-1])

    return G

def test():
    from matplotlib.pyplot import *

    fs = 3000.0
    w = 2 * np.pi * fs
    ts = 1.0/fs / 50.0
    N = 2048
    t = np.linspace(ts,float(N) * ts,N)
    x = np.sin(w * t) + np.cos(3*w*t)# + np.cos(5.2*w*t)*0.5 + np.cos(7.0*w*t)*0.2
    x = x + np.random.normal(size=x.shape[0]) * 0.5

    x = x * np.hanning(t.shape[0])

    #plot(t,x); show()

    yref = np.fft.fft(x)

    yref = yref[0:N/2]

    G = fftr(x,N/2)

    subplot(211, projection='polar')
    plot(np.angle(G),np.abs(G)); 
    plot(np.angle(yref),np.abs(yref)); 
    subplot(212)
    plot(np.abs(G))
    plot(np.abs(yref))
    show()

