function batchAnnotate(dirName, ext)

d = dir(fullfile(dirName,['*.' ext]));
files = {d.name};

for j = 1:length(files)
   
    wavName = fullfile(dirName, files{j});
    annoName = fullfile('~/Dropbox/Data/keyword/annotations', ['annotations_' strrep(files{j},'.wav','.csv')]);
    
    if ~exist(annoName, 'file')
        annotateWav(wavName, annoName);
        annotateKw(dirName, annoName);
    end
    
end
