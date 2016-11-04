function batchAnnotate(kw, dirName, outWavDir, clipDataFile, ext)
% batchAnnotate(kw, dirName, outWavDir, clipDataFile, ext)
%
% kw (string) - name of the keyword
% dirName (optional) - location of the input audio files
% outWavDIr (optional) - path the the output audio clip folder ([] if none)
% clipDataFile (optional) - path to the output .mat file containing info
%                           about the audio clips ([] if none)
% ext (optional) - file extention of the audio files
% 
% Example:
% batchAnnotate('okay_sense')

if ~exist('ext','var')
    ext = 'wav';
end
if ~exist('outWavDir','var')
    outWavDir = ['~/keyword/trainingWavs_' kw];
end
if ~exist('clipDataFile','var')
    clipDataFile = ['~/keyword/clipData_' kw '.mat'];
end
if ~exist('dirName','var')
    dirName = ['~/keyword/recordings/toAnnotate_' kw];
end

d = dir(fullfile(dirName,['*.' ext]));
files = {d.name};

[~,~] = mkdir(outWavDir);
annotationsDir = fullfile('~/keyword/annotations',kw);
[~,~] = mkdir(annotationsDir);
recordingsDir = '~/keyword/recordings';

for j = 1:length(files)
   
    wavName = fullfile(dirName, files{j});
    fileId = strrep(files{j},['.' ext],'');
    annoName = fullfile(annotationsDir,['annotations_' fileId '.csv']);
    
    if ~exist(annoName, 'file')
        annotateWav(wavName, annoName);
        annotateKw(dirName, annoName);
    
        copyfile(fullfile(dirName, files{j}), fullfile(recordingsDir,files{j}))

        if ~isempty(clipDataFile)
            getAnnotatedData(recordingsDir, annoName, clipDataFile);
            if ~isempty(outWavDir)
                createTrainingWavs(clipDataFile, outWavDir, kw, fileId);
            end
        end
    end
    
end
