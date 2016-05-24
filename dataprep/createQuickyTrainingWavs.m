function createQuickyTrainingWavs(dataFile, outDir)

load(dataFile) % -> kwClip kwRevClip speechClip backClip
[~,~] = mkdir(outDir);

Fs = 16000;

nKw = length(kwClip);
nKwRev = length(kwRevClip);
nSpeech = length(speechClip);
nBack = length(backClip);

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
for j = 1:nBack
    wav = resample(backClip{j},16,48);
    audiowrite(fullfile(outDir,['backClip_' num2str(j) '.wav']), wav, Fs)
end


%{
- Center each keyword in a 2-second window - clipped out of the ORIGINAL
file
- Also take the keyword portion and flip it for a negative
- Also take a random speech portion of the same duration as the keyword,
and replace
- 50 keywords, 50 reversed keywords, 200 non-keywords, 20 silences?
%}