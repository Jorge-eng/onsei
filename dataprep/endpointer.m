function [ptExt, ptInt] = endpointer(wav, th, fs)

if ~exist('fs','var')
    fs = 16000;
end
if ~exist('th','var')
    th = 0.01;
end

h = triang(round(0.1*fs)); 
h = h/sum(h);

env = sqrt(conv2(h, 1, wav.^2, 'same'));
act = env > th;

smoother = zeros(0.25*fs,1);
smoother(end/2:end) = 1;
smoother = smoother / sum(smoother);
forward = conv2(flipud(smoother), 1, single(act), 'same');
backward = conv2(smoother, 1, single(act), 'same');
act = act & (forward > 0.7 | backward > 0.7);

ptInt(1) = find(act, 1, 'first');
ptInt(2) = find(act, 1, 'last');

ptExt = ptInt + round(0.15*fs*[-1 1]);


