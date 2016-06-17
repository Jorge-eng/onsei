function createTrainingWavs(dataFile, outDir)
% createTrainingWavs(dataFile, outDir)

load(dataFile, 'info')
[~,~] = mkdir(outDir);

Fs = 16000;

vars = {'kwClip','kwRevClip','speechClip','backClip',...
        'earlyImplantClip','lateImplantClip',...
        'partialEarlyClip','partialLateClip',...
        'shiftEarlyClip','shiftLateClip'};
    
for v = 1:length(vars)
    var = vars{v};
    disp(['Creating ' var])
    data = load(dataFile, var);
    nFiles = length(data.(var));
    for n = 1:nFiles
        [~,fileTag] = fileparts(info.fileNames{n});
        for s = 1:length(data.(var){n})
            wav = resample(data.(var){n}{s},16,48);
            audiowrite(fullfile(outDir,[var '_' fileTag '_' num2str(s) '.wav']), wav, Fs)
        end
    end
end
