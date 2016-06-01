function getAnnotatedData2(dirName, csvFile, dataFile)

if ~exist('dataFile','var')
    dataFile = 'clipData2.mat';
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

kwClip = {};
kwRevClip = {};
backClip = {};
speechClip = {};
earlyImplantClip = {};
lateImplantClip = {};

Fs = 48000;
clipLen = 2 * Fs;
backPerFile = 2;
speechPerKw = 4;
for j = 1:length(fileNames)
    fileLoc = find(strcmp(fileName, fileNames{j}));
    fn = fullfile(dirName, fileNames{j});
    
    ii = find(strcmp('background', label(fileLoc)));
    background = [];
    for k = 1:length(ii)
        idx = fileLoc(ii(k));
        background = [background; audioread(fn, [clipStart(idx) clipEnd(idx)])];
    end
    
    ii = find(strcmp('speech', label(fileLoc)));
    speech = [];
    for k = 1:length(ii)
        idx = fileLoc(ii(k));
        speech = [speech; audioread(fn, [clipStart(idx) clipEnd(idx)])];
    end
    
    ii = find(strcmp('keyword', label(fileLoc)));
    for k = 1:length(ii)
        idx = fileLoc(ii(k));
        
        cs = clipStart(idx);
        ce = clipEnd(idx);
        cd = ce - cs;
        
        pad = floor((clipLen - cd) / 2);
        
        % keyword
        kw = audioread(fn, [cs-pad+1 cs-pad+clipLen]);
        kwClip{end+1} = kw;
        
        % reversed keyword
        x = kw;
        x(pad:pad+cd,:) = flipud(x(pad:pad+cd,:));
        kwRevClip{end+1} = x;
        
        % speech implants
        starts = 1 + floor(rand(1, speechPerKw)*(length(speech)-clipLen));
        for n = 1:speechPerKw
            x = kw;
            x(pad:pad+cd,:) = speech(starts(n):starts(n)+cd,:);
            speechClip{end+1} = x;
        end
    
        % partial speech implants
        kwMid = midPt(idx);
        % early
        x = kw;
        impInd = pad:kwMid;
        speechStart = 1 + floor(rand(1)*(length(speech)-length(impInd)));
        x(impInd,:) = speech(speechStart:speechStart+(length(impInd)-1),:);
        earlyImplantClip{end+1} = x;
        % late
        x = kw;
        impInd = kwMid+1:pad+cd;
        speechStart = 1 + floor(rand(1)*(length(speech)-length(impInd)));
        x(impInd,:) = speech(speechStart:speechStart+(length(impInd)-1),:);
        lateImplantClip{end+1} = x;
        
        % backgrounds
        interval = floor((length(background)-clipLen)/backPerFile);
        starts = 1:interval:interval*backPerFile;
        for n = 1:backPerFile
            backClip{end+1} = background(starts(n):(starts(n)-1)+clipLen,:);
        end
    end
    
end

info.fileName = fileName;
info.clipStart = clipStart;
info.clipEnd = clipEnd;
info.label = label;
save(dataFile,'kwClip','kwRevClip','speechClip','backClip','earlyImplantClip','lateImplantClip','info');
