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

noiseDir = '~/Dropbox/Data/keyword/noiseWavs';
d = dir(fullfile(noiseDir, '*.wav'));
noiseFiles = {d.name};
%noiseScales = 6:3:30;
noiseScales = Inf;

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
        
        % window for smooth-edge implants, same one for revKw and speech
        window = getWindow(cd+1, round(0.05*Fs), size(kw,2));
        
        kwVoc = applyRandomShift(kw, Fs);
        noise = getRandomNoise(noiseDir, noiseFiles, noiseScales, clipLen);
        kwClip{j}{end+1} = kwVoc + noise;
        
        % reversed keyword
        x = kwVoc;
        out = x(pad:pad+cd,:);
        in = flipud(x(pad:pad+cd,:));
        x(pad:pad+cd,:) = (1-window).*out + window.*in;
        noise = getRandomNoise(noiseDir, noiseFiles, noiseScales, clipLen);
        kwRevClip{j}{end+1} = x + noise;
        
        % speech implants
        starts = 1 + floor(rand(1, alignedSpeechPerKw)*(length(speech)-clipLen));
        for n = 1:alignedSpeechPerKw
            x = kw;
            out = x(pad:pad+cd,:);
            in = speech(starts(n):starts(n)+cd,:);
            x(pad:pad+cd,:) = (1-window).*out + window.*in;
            x = applyRandomShift(x, Fs);
            noise = getRandomNoise(noiseDir, noiseFiles, noiseScales, clipLen);
            speechClip{j}{end+1} = x + noise;
        end
        starts = 1 + floor(rand(1, randomSpeechPerKw)*(length(speech)-clipLen));
        for n = 1:randomSpeechPerKw
            x = speech(starts(n)+1:starts(n)+clipLen,:);
            x = applyRandomShift(x, Fs);
            noise = getRandomNoise(noiseDir, noiseFiles, noiseScales, clipLen);
            speechClip{j}{end+1} = x + noise;
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
        x = applyRandomShift(x, Fs);
        noise = getRandomNoise(noiseDir, noiseFiles, noiseScales, clipLen);
        earlyImplantClip{j}{end+1} = x + noise;
        
        % partial late - missing early part
        x = kw;
        backStart = 1 + floor(rand(1)*(length(background)-length(impInd)));
        in = background(backStart:backStart+(length(impInd)-1),:);
        x(impInd,:) = (1-window).*out + window.*in;
        x = applyRandomShift(x, Fs);
        noise = getRandomNoise(noiseDir, noiseFiles, noiseScales, clipLen);
        partialLateClip{j}{end+1} = x + noise;
        
        % partial implant late
        x = kw;
        impInd = pad+kwMid+1:pad+cd;
        window = getWindow(length(impInd), round(0.05*Fs), size(kw,2)); % smooth edge implants
        speechStart = 1 + floor(rand(1)*(length(speech)-length(impInd)));
        out = x(impInd,:);
        in = speech(speechStart:speechStart+(length(impInd)-1),:);
        x(impInd,:) = (1-window).*out + window.*in;
        x = applyRandomShift(x, Fs);
        noise = getRandomNoise(noiseDir, noiseFiles, noiseScales, clipLen);
        lateImplantClip{j}{end+1} = x + noise;
        
        % partial early -  missing late part
        x = kw;
        backStart = 1 + floor(rand(1)*(length(background)-length(impInd)));
        in = background(backStart:backStart+(length(impInd)-1),:);
        x(impInd,:) = (1-window).*out + window.*in;
        x = applyRandomShift(x, Fs);
        noise = getRandomNoise(noiseDir, noiseFiles, noiseScales, clipLen);
        partialEarlyClip{j}{end+1} = x + noise;
            
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
            kwSh = applyRandomShift(kwSh, Fs);
            noise = getRandomNoise(noiseDir, noiseFiles, noiseScales, clipLen);
            shiftLateClip{j}{end+1} = kwSh + noise;
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
            kwSh = applyRandomShift(kwSh, Fs);
            noise = getRandomNoise(noiseDir, noiseFiles, noiseScales, clipLen);
            shiftEarlyClip{j}{end+1} = kwSh + noise;
        end
            
    end
    
    % backgrounds
    interval = floor((length(background)-clipLen)/backPerFile);
    starts = 1:interval:interval*backPerFile;
    for n = 1:backPerFile
        noise = getRandomNoise(noiseDir, noiseFiles, noiseScales, clipLen);
        x = background(starts(n):(starts(n)-1)+clipLen,:);
        x = applyRandomShift(x, Fs);
        backClip{j}{end+1} = x + noise;
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
    
function noise = getRandomNoise(noiseDir, noiseFiles, noiseScales, clipLen)

noiseIdx = randperm(length(noiseFiles), 1);
scaleIdx = randperm(length(noiseScales), 1);

[noise, fs] = audioread(fullfile(noiseDir, noiseFiles{noiseIdx}));

startIdx = randperm(length(noise)-clipLen, 1);

noise = noise(startIdx+1:startIdx+clipLen,1);
noise = noise / noiseScales(scaleIdx);

noise = repmat(noise, [1 2]);

function kw = applyRandomShift(kw, Fs)
return % skip for now - it's causing overtraining

[clipLen, nCh] = size(kw);

shiftDenom = 16;
numJitter = 2;
shiftNum = randperm(2*numJitter+1, 1) - (numJitter+1) + shiftDenom;

clip = kw;%(pad+1:pad+cd, :);

clipShift = shiftAndStretch(clip(:,1), Fs, 1.0, shiftNum, shiftDenom);
clipShift = repmat(clipShift, [1 nCh]);
lenOut = size(clipShift, 1);

gain = sqrt(mean(clip(:,1).^2)) / sqrt(mean(clipShift(:,1).^2));
clipShift = clipShift * gain;

lenDiff = clipLen - lenOut;
padIn = floor(lenDiff/2);
in = clip(padIn+1:padIn+lenOut,:);

window = getWindow(lenOut, round(0.05*Fs), nCh);
clipShift = (1-window).*in + window.*clipShift;

kw(padIn+1:padIn+lenOut,:) = clipShift;
