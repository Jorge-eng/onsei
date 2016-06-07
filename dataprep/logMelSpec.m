function logM = logMelSpec(wav, Fs, nMel, windowLen_s, frameShift_s)
% logM = logMelSpec(wav, Fs, nMel, windowLen_s, frameShift_s)

if ~exist('nMel','var')
    nMel = 40;
end
if ~exist('windowLen_s','var')
    windowLen_s = 0.025;
end
if ~exist('frameShift_s','var')
    frameShift_s = 0.010;
end

windowLen = Fs * windowLen_s;
WINDOW = hann(windowLen);
NFFT = 2^nextpow2(windowLen);
NOVERLAP = windowLen - (Fs * frameShift_s);
frameRate = Fs / NOVERLAP;

[S,F] = spectrogram(wav(:,1),WINDOW,NOVERLAP,NFFT,Fs);
nFreq = length(F);

mf = melfilter(nMel, F);
M = mf * abs(S).^2;
logM = log(M);
