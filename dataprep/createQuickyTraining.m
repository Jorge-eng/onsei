function createQuickyTraining(dataFile)

load(dataFile)

Fs = 16000;
windowLen_s = 0.025;
frameShift_s = 0.010;
nMel = 40;

nKw = length(kwClip);
nKwRev = length(kwRevClip);
nSpeech = length(speechClip);
nBack = length(backClip);

for j = 1:nKw
    kwClip{j} = resample(kwClip{j},16,48);
    kwSpec{j} = logMelSpec(kwClip{j}, Fs, nMel, windowLen_s, frameShift_s);
end
for j = 1:nKwRef
    kwRevClip{j} = resample(kwRevClip{j},16,48);
    kwRevSpec{j} = logMelSpec(kwRevClip{j}, Fs, nMel, windowLen_s, frameShift_s);
end
for j = 1:nSpeech
    speechClip{j} = resample(speechClip{j},16,48);
    speechSpec{j} = logMelSpec(speechClip{j}, Fs, nMel, windowLen_s, frameShift_s);
end
for j = 1:nBack
    backClip{j} = resample(backClip{j},16,48);
    backSpec{j} = logMelSpec(backClip{j}, Fs, nMel, windowLen_s, frameShift_s);
end

features = cat(3, kwSpec{:}, kwRevSpec{:}, speechSpec{:}, backSpec{:});
labels = [ones(nKw, 1, 'single'); zeros(nKwRev+nSpeech+nBack,1,'single')];

save(dataFile, 'kwSpec','kwRevSpec','speechSpec','backSpec', '-append')
save(dataFile, 'features', 'labels', '-append')

%{
- Center each keyword in a 2-second window - clipped out of the ORIGINAL
file
- Also take the keyword portion and flip it for a negative
- Also take a random speech portion of the same duration as the keyword,
and replace
- 50 keywords, 50 reversed keywords, 200 non-keywords, 20 silences?
%}