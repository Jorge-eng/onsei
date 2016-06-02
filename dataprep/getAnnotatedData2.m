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
backPerFile = 4;
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
        
        % stretched keyword
        
        % pitch-shifted keyword
        
        % noise-added keyword
        
        % window for smooth-edge implants, same one for revKw and speech
        window = getWindow(cd+1, round(0.05*Fs), size(kw,2));
        
        % reversed keyword
        x = kw;
        out = x(pad:pad+cd,:);
        in = flipud(x(pad:pad+cd,:));
        x(pad:pad+cd,:) = (1-window).*out + window.*in;
        kwRevClip{end+1} = x;
        
        % speech implants
        starts = 1 + floor(rand(1, speechPerKw)*(length(speech)-clipLen));
        for n = 1:speechPerKw
            x = kw;
            out = x(pad:pad+cd,:);
            in = speech(starts(n):starts(n)+cd,:);
            x(pad:pad+cd,:) = (1-window).*out + window.*in;
            speechClip{end+1} = x;
        end
    
        % partial speech implants
        kwMid = midPt(idx);
        % early
        x = kw;
        impInd = pad:kwMid;
        window = getWindow(length(impInd), round(0.05*Fs), size(kw,2)); % smooth edge implants
        speechStart = 1 + floor(rand(1)*(length(speech)-length(impInd)));
        out = x(impInd,:);
        in = speech(speechStart:speechStart+(length(impInd)-1),:);
        x(impInd,:) = (1-window).*out + window.*in;
        earlyImplantClip{end+1} = x;
        % late
        x = kw;
        impInd = kwMid+1:pad+cd;
        window = getWindow(length(impInd), round(0.05*Fs), size(kw,2)); % smooth edge implants
        speechStart = 1 + floor(rand(1)*(length(speech)-length(impInd)));
        out = x(impInd,:);
        in = speech(speechStart:speechStart+(length(impInd)-1),:);
        x(impInd,:) = (1-window).*out + window.*in;
        lateImplantClip{end+1} = x;
        
    end
    
    % backgrounds
    interval = floor((length(background)-clipLen)/backPerFile);
    starts = 1:interval:interval*backPerFile;
    for n = 1:backPerFile
        backClip{end+1} = background(starts(n):(starts(n)-1)+clipLen,:);
    end
    
end

info.fileName = fileName;
info.clipStart = clipStart;
info.clipEnd = clipEnd;
info.label = label;
save(dataFile,'kwClip','kwRevClip','speechClip','backClip','earlyImplantClip','lateImplantClip','info');

function window = getWindow(winDur, rampDur, numChan)
% window function

%rampDur = round(0.05*Fs);
ramp = hanning(rampDur);
ramp = ramp(1:floor(end/2));
ramp = repmat(ramp, [1 numChan]);
window = ones(winDur, numChan);
window(1:length(ramp),:) = ramp;
window(end-length(ramp)+1:end,:) = flipud(ramp);
        
