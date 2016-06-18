function out = shiftAndStretch(in, Fs, stretchRatio, shiftNum, shiftDenom)
% out = shiftAndStretch(in, Fs, stretchRatio, shiftNum, shiftDenom)

% if we want to shift up, we stretch and the downsample by the inverse amount
% if we want to shift up and stretch, we stretch even more before the
% inverse

amountToStretch = stretchRatio * shiftNum / shiftDenom;

fftSize = 1024*Fs/16e3;
%padSize = round(fftSize / 4 / amountToStretch);
%in = [zeros(padSize, 1); in; zeros(padSize,1)];

if amountToStretch ~= 1
    out = pvoc(in, amountToStretch, fftSize); 
else
    out = in;
end

if shiftNum ~= shiftDenom
    
    out = resample(out, shiftNum, shiftDenom);  

end
