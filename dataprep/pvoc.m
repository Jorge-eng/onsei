function y = pvoc(x, r, nfft)
% y = pvoc(x, r, nfft)  Time-scale a signal to r times faster with phase vocoder

if nargin < 3
  nfft = 1024;
end

hop = nfft/4;

X = stft(x', nfft, nfft, hop);

% Calculate the new timebase samples
[rows, cols] = size(X);
t = 0:r:(cols-2);

% Generate the new spectrogram
X2 = pvsample(X, t, hop);

% Invert to a waveform
y = istft(X2, nfft, nfft, hop)';
