function createQuickyTrainingWavs(dataFile, outDir)

%load(dataFile) % -> kwClip kwRevClip speechClip backClip earlyImplantClip lateImplantClip
[~,~] = mkdir(outDir);

Fs = 16000;

vars = {'kwClip','kwRevClip','speechClip','backClip',...
        'earlyImplantClip','lateImplantClip',...
        'partialEarlyClip','partialLateClip',...
        'shiftEarlyClip','shiftLateClip'};
    
for j = 1:length(vars)
    data = load(dataFile, vars{j});
    N = length(data.(vars{j}));
    for n = 1:N
        wav = resample(data.(vars{j}){n},16,48);
        audiowrite(fullfile(outDir,[vars{j} '_' num2str(n) '.wav']), wav, Fs)
    end
end

%{
- Center each keyword in a 2-second window - clipped out of the ORIGINAL
file
- Also take the keyword portion and flip it for a negative
- Also take a random speech portion of the same duration as the keyword,
and replace
- 50 keywords, 50 reversed keywords, 200 non-keywords, 20 silences?
%}