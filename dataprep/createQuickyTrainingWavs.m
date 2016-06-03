function createQuickyTrainingWavs(dataFile, outDir)

load(dataFile) % -> kwClip kwRevClip speechClip backClip earlyImplantClip lateImplantClip
[~,~] = mkdir(outDir);

Fs = 16000;

nKw = length(kwClip);
nKwRev = length(kwRevClip);
nSpeech = length(speechClip);
nBack = length(backClip);
nImpEarly = length(earlyImplantClip);
nImpLate = length(lateImplantClip);
nPartEarly = length(partialEarlyClip);
nPartLate = length(partialLateClip);

for j = 1:nKw
    wav = resample(kwClip{j},16,48);
    audiowrite(fullfile(outDir,['kwClip_' num2str(j) '.wav']), wav, Fs)
end
for j = 1:nKwRev
    wav = resample(kwRevClip{j},16,48);
    audiowrite(fullfile(outDir,['kwRevClip_' num2str(j) '.wav']), wav, Fs)
end
for j = 1:nSpeech
    wav = resample(speechClip{j},16,48);
    audiowrite(fullfile(outDir,['speechClip_' num2str(j) '.wav']), wav, Fs)
end
for j = 1:nImpEarly
    wav = resample(earlyImplantClip{j},16,48);
    audiowrite(fullfile(outDir,['earlyImplantClip_' num2str(j) '.wav']), wav, Fs)
end
for j = 1:nImpLate
    wav = resample(lateImplantClip{j},16,48);
    audiowrite(fullfile(outDir,['lateImplantClip_' num2str(j) '.wav']), wav, Fs)
end
for j = 1:nBack
    wav = resample(backClip{j},16,48);
    audiowrite(fullfile(outDir,['backClip_' num2str(j) '.wav']), wav, Fs)
end
for j = 1:nPartEarly
    wav = resample(partialEarlyClip{j},16,48);
    audiowrite(fullfile(outDir,['partialEarlyClip_' num2str(j) '.wav']), wav, Fs)
end
for j = 1:nPartLate
    wav = resample(partialLateClip{j},16,48);
    audiowrite(fullfile(outDir,['partialLateClip_' num2str(j) '.wav']), wav, Fs)
end

%{
- Center each keyword in a 2-second window - clipped out of the ORIGINAL
file
- Also take the keyword portion and flip it for a negative
- Also take a random speech portion of the same duration as the keyword,
and replace
- 50 keywords, 50 reversed keywords, 200 non-keywords, 20 silences?
%}