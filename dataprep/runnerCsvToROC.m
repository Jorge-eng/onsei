function [num, tot, loc] = runnerCsvToROC(dirName, ths, counts, fileType)

if ~exist('ths','var')
    nTh = 20;
    ths = linspace(4095/nTh,4095,nTh);
end
if ~exist('counts','var')
    counts = 1:20;
end
if ~exist('fileType','var')
    fileType = 'csv';
end

%dirName = '';
d = dir(fullfile(dirName, ['*.' fileType]));

for j = 1:length(d)
    disp(d(j).name)
    if strcmp(fileType,'csv')
        data = csvread(fullfile(dirName, d(j).name));
    elseif strcmp(fileType,'mat')
        data = load(fullfile(dirName, d(j).name));
        data = squeeze(data.prob);
    end
    tot(j) = size(data,1);
    for m = 1:length(counts)
        h = ones(counts(m), 1, 'single');
        for n = 1:size(ths,2)
            [num(j,m,n,:), loc(j,m,n,:)] = runnerDetector(data(:,2:end), h, ths(:,n));
        end
    end
end
