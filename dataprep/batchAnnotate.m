function batchAnnotate(dirName, ext, kw, clipDataFile, outWavDir)
% batchAnnotate(dirName, ext, kw, clipDataFile, outWavDir)
%
% Example:
% batchAnnotate('~/Dropbox/Data/keyword/recordings/toAnnotateSense','wav','okay_sense','~/Dropbox/Data/keyword/clipData_irs.mat','~/Dropbox/Data/keyword/trainingWavsIrs')

d = dir(fullfile(dirName,['*.' ext]));
files = {d.name};

for j = 1:length(files)
   
    wavName = fullfile(dirName, files{j});
    fileId = strrep(files{j},'.wav','');
    [~,~] = mkdir(fullfile('~/Dropbox/Data/keyword/annotations',kw));
    annoName = fullfile('~/Dropbox/Data/keyword/annotations',kw,['annotations_' fileId '.csv']);
    
    if ~exist(annoName, 'file')
        annotateWav(wavName, annoName);
        annotateKw(dirName, annoName);
    
        copyfile(fullfile(dirName, files{j}), fullfile('~/Dropbox/Data/keyword/recordings',files{j}))

        if exist('clipDataFile','var')
            getAnnotatedData('~/Dropbox/Data/keyword/recordings', annoName, clipDataFile);
            if exist('outWavDir','var')
                createTrainingWavs(clipDataFile, outWavDir, fileId);
            end
        end
    end
    
end
