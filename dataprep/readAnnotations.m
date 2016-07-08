function [fileNames, fileName, clipStart, clipEnd, label, midPt] = readAnnotations(csvFile)
%[fileNames, fileName, clipStart, clipEnd, label, midPt] = readAnnotations(csvFile)

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
    
    line = strrep(line,'^M',''); % Where are these coming from?
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
