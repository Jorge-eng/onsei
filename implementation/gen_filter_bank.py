#!/usr/bin/python
import numpy as np
def hz2mel(hz):
    """Convert a value in Hertz to Mels
    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1+hz/700.0)
    
def mel2hz(mel):
    """Convert a value in Mels to Hertz
    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700*(10**(mel/2595.0)-1)

def get_filterbanks(nfilt=20,nfft=512,samplerate=16000,lowfreq=0,highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"
    
    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel,highmel,nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft+1)*mel2hz(melpoints)/samplerate)

    fbank = np.zeros([nfilt,nfft/2+1])
    for j in xrange(0,nfilt):
        for i in xrange(int(bin[j]),int(bin[j+1])):
            fbank[j,i] = (i - bin[j])/(bin[j+1]-bin[j])
        for i in xrange(int(bin[j+1]),int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i)/(bin[j+2]-bin[j+1])
    return fbank               

if __name__ == '__main__':
    fs = 16000.
    winlen = 0.025
    nfft = np.int(np.power(2, np.ceil(np.log2(winlen*fs))))
    nfilt = 40

    print 'fft_size = %d' % nfft
    fbank = get_filterbanks(nfilt,nfft,fs)
    print fbank.shape

    coeffs = []
    indices = []
    pairs = []
    for irow in range(fbank.shape[0]):
        b = np.where(fbank[irow])
        c = fbank[irow][b]
        istart = b[0][0]
        iend = b[0][-1]

        indices.extend(b[0].astype(str).tolist())
        coeffs.extend( (c * 32767).astype(int).astype(str).tolist())
        pairs.append((istart,iend))

    strpairs = ['{%d,%d}' % pair for pair in pairs]
    N = len(indices)
    
    print 'const static uint8_t k_indices[%d] = {%s};\n' % (N,','.join(indices))
    print 'const static int16_t k_coeffs[%d] = {%s};\n' % (N,','.join(coeffs))
    print 'const static uint8_t k_fft_index_pairs[%d][%d] = {%s}' % (nfilt,2,','.join(strpairs))

        

