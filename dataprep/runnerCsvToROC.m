function [num, tot, loc] = runnerCsvToROC(dirName, ths, counts)

if ~exist('ths','var')
    nTh = 20;
    ths = linspace(4095/nTh,4095,nTh);
end
if ~exist('counts','var')
    counts = 1:20;
end

%dirName = '';
d = dir(fullfile(dirName, '*.csv'));

for j = 1:length(d)
    disp(d(j).name)
    data = csvread(fullfile(dirName, d(j).name));
    tot(j) = size(data,1);
    for m = 1:length(counts)
        h = ones(counts(m), 1, 'single');
        for n = 1:length(ths)
            [num(j,m,n,:), loc(j,m,n,:)] = runnerDetector(data(:,2:4), h, ths(n));
        end
    end
end
