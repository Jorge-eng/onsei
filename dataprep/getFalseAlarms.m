load ../net/prob_long.mat
outDir = '~/Dropbox/Data/keyword/falseAlarms';
th = 0.5;
ii = find(prob(:,2) > th);
p = prob(ii,2);

fs = 16e3;
frameShift = 0.01 * fs;
winLen = 1.6 * fs;

startSamps = 1 + frameShift*startTimes(ii);
endSamps = startSamps + winLen - 1;

files = cellstr(files);
for j = 1:length(ii)
    idx = ii(j);
    file = files{1+fileIdx(idx)};
    clip = audioread(file, double([startSamps(j) endSamps(j)]));
    
    outName = ['speechClip_fa_' num2str(j) '.wav'];
    audiowrite(fullfile(outDir, outName), clip, fs)
    %if file ~= lastfile
    %    wav = audioread(file);
    %end
    %lastfile = file;
    
end