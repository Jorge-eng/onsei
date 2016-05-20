function logM = audioStreamFeatures(wavFiles)

if ~iscell(wavFiles)
    wavFiles = {wavFiles};
end

Fs = 16000;
windowLen_s = 0.025;
frameShift_s = 0.010;
nMel = 40;

for j = 1:length(wavFiles)
    wav = audioread(wavFiles{j});
    wav = resample(wav(:,1), 16, 48);
    logM{j} = logMelSpec(wav, Fs, nMel, windowLen_s, frameShift_s);    
end

if length(logM) == 1
    logM = logM{1};
end
