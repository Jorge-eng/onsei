function batchAnnotate(dirName, ext, clipDataFile, outWavDir)
% batchAnnotate(dirName, ext, clipDataFile, outWavDir)

d = dir(fullfile(dirName,['*.' ext]));
files = {d.name};

for j = 1:length(files)
   
    wavName = fullfile(dirName, files{j});
    fileId = strrep(files{j},'.wav','');
    annoName = fullfile('~/Dropbox/Data/keyword/annotations', ['annotations_' fileId '.csv']);
    
    if ~exist(annoName, 'file')
        annotateWav(wavName, annoName);
        annotateKw(dirName, annoName);
        
        if exist('clipDataFile','var')
            getAnnotatedData('~/Dropbox/Data/keyword/recordings', annoName, clipDataFile);
            if exist('outWavDir','var')
                createTrainingWavs(clipDataFile, outWavDir, fileId);
            end
        end
    end
    
end
