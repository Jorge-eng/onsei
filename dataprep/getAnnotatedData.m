function getAnnotatedData(dirName, csvFile, dataFile)
% getAnnotatedData(dirName, csvFile, dataFile)

if ~exist('dataFile','var')
    dataFile = 'clipData.mat';
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

for j = 1:length(fileNames)
    kwClip{j} = {};
    kwRevClip{j} = {};
    backClip{j} = {};
    speechClip{j} = {};
    earlyImplantClip{j} = {};
    lateImplantClip{j} = {};
    partialEarlyClip{j} = {};
    partialLateClip{j} = {};
    shiftEarlyClip{j} = {};
    shiftLateClip{j} = {};
end

Fs = 48000;
clipLen = 2 * Fs;
backPerFile = 4;
alignedSpeechPerKw = 2;
randomSpeechPerKw = 2;

for j = 1:length(fileNames)
    disp(['File ' num2str(j) ' out of ' num2str(length(fileNames))])
    fileLoc = find(strcmp(fileName, fileNames{j}));
    fn = fullfile(dirName, fileNames{j});
    auInfo = audioinfo(fn);
    
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
        
        if clipLen - cd > cs
            disp('Discarding first keyword')
            continue
        end
        pad = clipLen - cd;
        
        % keyword
        kw = audioread(fn, [cs-pad+1 cs-pad+clipLen]);
        kwClip{j}{end+1} = kw;
        
        % stretched keyword
        
        % pitch-shifted keyword
        %kwUp = shiftAndStretch(kw(:,1), Fs, 1, 9, 8);
        
        % noise-added keyword
        
        % window for smooth-edge implants, same one for revKw and speech
        window = getWindow(cd+1, round(0.05*Fs), size(kw,2));
        
        % reversed keyword
        x = kw;
        out = x(pad:pad+cd,:);
        in = flipud(x(pad:pad+cd,:));
        x(pad:pad+cd,:) = (1-window).*out + window.*in;
        kwRevClip{j}{end+1} = x;
        
        % speech implants
        starts = 1 + floor(rand(1, alignedSpeechPerKw)*(length(speech)-clipLen));
        for n = 1:alignedSpeechPerKw
            x = kw;
            out = x(pad:pad+cd,:);
            in = speech(starts(n):starts(n)+cd,:);
            x(pad:pad+cd,:) = (1-window).*out + window.*in;
            speechClip{j}{end+1} = x;
        end
        starts = 1 + floor(rand(1, randomSpeechPerKw)*(length(speech)-clipLen));
        for n = 1:randomSpeechPerKw
            x = speech(starts(n):starts(n)+clipLen,:);
            speechClip{j}{end+1} = x;
        end
        
        % partials --- 
        kwMid = midPt(idx);
        
        % partial implant early
        x = kw;
        impInd = pad:pad+kwMid;
        window = getWindow(length(impInd), round(0.05*Fs), size(kw,2)); % smooth edge implants
        speechStart = 1 + floor(rand(1)*(length(speech)-length(impInd)));
        out = x(impInd,:);
        in = speech(speechStart:speechStart+(length(impInd)-1),:);
        x(impInd,:) = (1-window).*out + window.*in;
        earlyImplantClip{j}{end+1} = x;
        
        % partial late - missing early part
        backStart = 1 + floor(rand(1)*(length(background)-length(impInd)));
        in = background(backStart:backStart+(length(impInd)-1),:);
        x(impInd,:) = (1-window).*out + window.*in;
        partialLateClip{j}{end+1} = x;
        
        % partial implant late
        x = kw;
        impInd = pad+kwMid+1:pad+cd;
        window = getWindow(length(impInd), round(0.05*Fs), size(kw,2)); % smooth edge implants
        speechStart = 1 + floor(rand(1)*(length(speech)-length(impInd)));
        out = x(impInd,:);
        in = speech(speechStart:speechStart+(length(impInd)-1),:);
        x(impInd,:) = (1-window).*out + window.*in;
        lateImplantClip{j}{end+1} = x;
        
        % partial early -  missing late part
        backStart = 1 + floor(rand(1)*(length(background)-length(impInd)));
        in = background(backStart:backStart+(length(impInd)-1),:);
        x(impInd,:) = (1-window).*out + window.*in;
        partialEarlyClip{j}{end+1} = x;
            
        % Partials shifted to boundaries --
        % The end
        if pad + (cd-kwMid) < cs
            kwSh = audioread(fn, [cs-(pad+cd-kwMid)+1 cs-(pad+cd-kwMid)+clipLen]);
            impInd = 1:cd-kwMid;
            window = getWindow(length(impInd), round(0.05*Fs), size(kw,2));
            out = kwSh(impInd,:);
            backStart = 1 + floor(rand(1)*(length(background)-length(impInd)));
            in = background(backStart:backStart+(length(impInd)-1),:);
            kwSh(impInd,:) = (1-window).*out + window.*in;
            shiftLateClip{j}{end+1} = kwSh;
        end
        % The beginning
        if cs+kwMid+clipLen <= auInfo.TotalSamples
            kwSh = audioread(fn, [cs+kwMid+1 cs+kwMid+clipLen]);
            impInd = clipLen-kwMid+1:clipLen;
            window = getWindow(length(impInd), round(0.05*Fs), size(kw,2));
            out = kwSh(impInd,:);
            backStart = 1 + floor(rand(1)*(length(background)-length(impInd)));
            in = background(backStart:backStart+(length(impInd)-1),:);
            kwSh(impInd,:) = (1-window).*out + window.*in;
            shiftEarlyClip{j}{end+1} = kwSh;
        end
            
    end
    
    % backgrounds
    interval = floor((length(background)-clipLen)/backPerFile);
    starts = 1:interval:interval*backPerFile;
    for n = 1:backPerFile
        backClip{j}{end+1} = background(starts(n):(starts(n)-1)+clipLen,:);
    end
    
end

disp('Saving Data...')
info.fileNames = fileNames;
info.fileName = fileName;
info.clipStart = clipStart;
info.clipEnd = clipEnd;
info.label = label;
save(dataFile,'*Clip','info')

function window = getWindow(winDur, rampDur, numChan)
% window function

%rampDur = round(0.05*Fs);
ramp = hanning(rampDur);
ramp = ramp(1:floor(end/2));
ramp = repmat(ramp, [1 numChan]);
window = ones(winDur, numChan);
window(1:length(ramp),:) = ramp;
window(end-length(ramp)+1:end,:) = flipud(ramp);
        