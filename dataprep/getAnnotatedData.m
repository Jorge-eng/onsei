function [data, labelNames, info] = getAnnotatedData(dirName, csvFile)

fileName = {};
clipStart = [];
clipEnd = [];
label = {};

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
end
fclose(fid);

labelNames = unique(label);
data = cell(length(labelNames),1);
for j = 1:length(labelNames)

    labelLoc = find(strcmp(label, labelNames{j}));
    for k = 1:length(labelLoc)
        ii = labelLoc(k);
        data{j}{end+1} = audioread(fullfile(dirName, fileName{ii}), [clipStart(ii) clipEnd(ii)]);
    end
    
end

info.fileName = fileName;
info.clipStart = clipStart;
info.clipEnd = clipEnd;
info.label = label;
