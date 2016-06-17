function [levels, pow, fileNames] = getLevels(dirName, csvFile, calibrate)
% [levels, pow, fileNames] = getLevels(dirName, csvFile, calibrate)

if ~exist('calibrate','var')
    calibrate = false;
end

fileName = {};
clipStart = [];
clipEnd = [];
label = {};
midPt = [];

fid = fopen(csvFile,'r');
while 1
    line = fgetl(fid);
    if ~ischar(line)
        break
    end
    
    line = strsplit(line,',');
    
    fileName{end+1} = line{1};
    clipStart(end+1) = str2double(line{2});
    clipEnd(end+1) = str2double(line{3});
    label{end+1} = line{4};
    if strcmp('keyword',label{end})
        midPt(end+1) = str2double(line{5});
    else
        midPt(end+1) = NaN;
    end
end
fclose(fid);

fileNames = unique(fileName);

Fs = 48000;
[b, a] = butter(5, 150/Fs*2, 'high');

for j = 1:length(fileNames)
    disp(['File ' num2str(j) ' out of ' num2str(length(fileNames))])
    fileLoc = find(strcmp(fileName, fileNames{j}));
    fn = fullfile(dirName, fileNames{j});
        
    ii = find(strcmp('keyword', label(fileLoc)));
    for k = 1:length(ii)
        idx = fileLoc(ii(k));
        
        cs = clipStart(idx);
        ce = clipEnd(idx);
        
        % keyword
        kw = audioread(fn, [cs ce]);
        kwFilt = filter(b, a, kw);
        pow(j, k) = mean(kwFilt(:,1).^2);
            
    end
    
    valid = find(pow(j,:) > 0);
    levels(j) = sqrt(mean(pow(j,valid)));

end

if calibrate
    gains = mean(levels) ./ levels;
    
    for j = 1:length(fileNames)
        
        fn = fullfile(dirName, fileNames{j});
        [wav, Fs] = audioread(fn);
        wav = wav*gains(j);
        audiowrite(fn, wav, Fs)
        
    end
end
