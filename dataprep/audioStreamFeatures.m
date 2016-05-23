function [features, logM] = audioStreamFeatures(wavFiles)

if ~iscell(wavFiles)
    wavFiles = {wavFiles};
end

Fs = 16000;
windowLen_s = 0.025;
frameShift_s = 0.010;
nMel = 40;

detWinLen = 198; 
detWinShift = 20;

for j = 1:length(wavFiles)
    wav = audioread(wavFiles{j});
    wav = resample(wav(:,1), 16, 48);
    logM{j} = logMelSpec(wav, Fs, nMel, windowLen_s, frameShift_s);    
    
    N = size(logM{j}, 2);
    starts = 1:detWinShift:N-detWinLen;
    features{j} = zeros(nMel,detWinLen,length(starts),'single');
    for k = 1:length(starts)
        features{j}(:,:,k) = logM{j}(:,starts(k)+(0:detWinLen-1));
    end

end

if length(logM) == 1
    logM = logM{1};
    features = features{1};
end
