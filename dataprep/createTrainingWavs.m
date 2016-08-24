function createTrainingWavs(dataFile, outDir, kw, fileId)
% createTrainingWavs(dataFile, outDir, kw, fileId)

[~,~] = mkdir(outDir);
[~,~] = mkdir(fullfile(outDir,'annotations'));

Fs = 16000;

if ~exist('kw','var')
    kw = '';
end
if ~exist('fileId','var')
    d = whos('-file',dataFile,'file_*');
    fileVars = {d.name};
else
    fileVars = {['file_' fileId]};
end

vars = {'kwClip','kwRevClip','backClip',...
        'speechAlignedClip','speechRandomClip',...
        'earlyImplantClip','lateImplantClip',...
        'partialEarlyClip','partialLateClip',...
        'shiftEarlyClip','shiftLateClip'};

load(dataFile, 'info')

for f = 1:length(fileVars)
    disp(['Loading ' fileVars{f}])
    fileTag = strrep(fileVars{f},'file_','');
    annoVar = ['annotation_' fileTag];
    
    d = load(dataFile, fileVars{f});
    audioData = d.(fileVars{f});
    
    d = load(dataFile, annoVar);
    annoData = d.(annoVar);
    
    for v = 1:length(vars)
        var = vars{v};
        disp(['Creating ' var])
        
        for s = 1:length(audioData.(var))
            %wav = resample(data.(var){n}{s},16,48);
            wav = audioData.(var){s};
            anno = annoData.(var){s};
    
            wav = 1.0 * wav; % Avoid clipping
            fileRoot = fullfile(outDir,[var '_' kw '_' fileTag '_' num2str(s)]);
            audiowrite([fileRoot '.wav'], wav, Fs)
            csvwrite([fileRoot '.csv'], anno)
        end
        
    end
end
