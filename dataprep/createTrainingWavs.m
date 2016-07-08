function createTrainingWavs(dataFile, outDir, fileId)
% createTrainingWavs(dataFile, outDir)

[~,~] = mkdir(outDir);

Fs = 16000;

if ~exist('fileId','var')
    d = whos('-file',dataFile,'file_*');
    fileVars = {d.name};
else
    fileVars = {['file_' fileId]};
end

vars = {'kwClip','kwRevClip','speechClip','backClip',...
        'earlyImplantClip','lateImplantClip',...
        'partialEarlyClip','partialLateClip',...
        'shiftEarlyClip','shiftLateClip'};

load(dataFile, 'info')

for f = 1:length(fileVars)
    disp(['Loading ' fileVars{f}])
    fileTag = strrep(fileVars{f},'file_','');
    
    d = load(dataFile, fileVars{f});
    data = d.(fileVars{f});
    
    for v = 1:length(vars)
        var = vars{v};
        disp(['Creating ' var])
        
        for s = 1:length(data.(var))
            %wav = resample(data.(var){n}{s},16,48);
            wav = data.(var){s};
            wav = 1.0 * wav; % Avoid clipping
            audiowrite(fullfile(outDir,[var '_' fileTag '_' num2str(s) '.wav']), wav, Fs)
        end
        
    end
end
