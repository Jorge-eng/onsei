function [num, tot] = runnerCsvToROC(dirName, ths, counts)

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
            num(j,m,n,:) = getDetCounts(data(:,2:4), h, ths(n));
        end
    end
end

function num = getDetCounts(data, h, th)

for kw = 1:size(data, 2)
    det = single(data(:,kw) > th);
    det = conv2(h, 1, det, 'same');
    det = single(det==length(h));
    det = diff(det) > 0;
    num(kw) = length(find(det));
end
