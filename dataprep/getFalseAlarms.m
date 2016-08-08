kw = 'snooze';
iter = '22';
inTag = 'iter';
outTag = 'iter2';
winLen_s = 1.6;
load(['../net/prob_long_' kw '_' inTag '_' iter '.mat'])
outDir = ['~/Dropbox/Data/keyword/falseAlarms_' kw '_' iter];
[~,~] = mkdir(outDir);
th = 0.5;
ii = find(prob(:,2) > th);
p = prob(ii,2);

fs = 16e3;
frameShift = 0.01 * fs;
winLen = winLen_s * fs;

startSamps = 1 + frameShift*startTimes(ii);
endSamps = startSamps + winLen - 1;

files = cellstr(files);
for j = 1:length(ii), j
    idx = ii(j);
    file = files{1+fileIdx(idx)};
    clip = audioread(file, double([startSamps(j) endSamps(j)]));
    
    outName = ['speechClip_' kw '_fa_' outTag '_' num2str(j) '.wav'];
    audiowrite(fullfile(outDir, outName), clip, fs)
    %if file ~= lastfile
    %    wav = audioread(file);
    %end
    %lastfile = file;
    
end