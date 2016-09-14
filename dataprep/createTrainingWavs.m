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
        if ~isfield(audioData, var)
            continue
        end
        
        disp(['Creating ' var])
        
        for s = 1:length(audioData.(var))
            %wav = resample(data.(var){n}{s},16,48);
            wav = audioData.(var){s};
            anno = annoData.(var){s};
             
            wav = 1.0 * wav; % Avoid clipping
            fileRoot = fullfile(outDir,[var '_' kw '_' fileTag '_' num2str(s)]);
            audiowrite([fileRoot '.wav'], wav, Fs)
            
            annowrite([fileRoot '.csv'], anno)
        end
        
    end
end

function annowrite(fileName, anno)

if iscell(anno)
    temp = '';
    for ev = 1:length(anno)
        temp = [temp ',' anno{ev}{1} ',' num2str(anno{ev}{2}(1)) ',' num2str(anno{ev}{2}(2))];
    end
    anno = temp(2:end);
    fid = fopen(fileName, 'w');
    fprintf(fid, '%s', anno);
    fclose(fid);
elseif isnumeric(anno)
    csvwrite(fileName, anno)
end
