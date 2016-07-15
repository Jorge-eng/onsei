function getAnnotatedData(dirName, csvFile, dataFile)
% getAnnotatedData(dirName, csvFile, dataFile)

if ~exist('dataFile','var')
    dataFile = 'clipData.mat';
end

[fileNames, fileName, clipStart, clipEnd, label, midPt] = readAnnotations(csvFile);

info.fileNames = fileNames;
info.fileName = fileName;
info.clipStart = clipStart;
info.clipEnd = clipEnd;
info.label = label;
save(dataFile,'info')

clipLen_s = 1.6;
backPerFile = 4;
alignedSpeechPerKw = 2;
randomSpeechPerKw = 2;

noiseDir = '~/Dropbox/Data/keyword/noiseWavs';
d = dir(fullfile(noiseDir, '*.wav'));
noiseFiles = {d.name};
noiseScales = 1./10.^((-18:-2:-30)/20);
%noiseScales = Inf;

irDir = 'IR';
d = dir(fullfile(irDir,'*16k.wav'));
irFiles = {d.name};

for j = 1:length(fileNames)
    disp(['File ' num2str(j) ' out of ' num2str(length(fileNames))])
    fileLoc = find(strcmp(fileName, fileNames{j}));
    fn = fullfile(dirName, fileNames{j});
    auInfo = audioinfo(fn);
    FsIn = round(auInfo.SampleRate/1000)*1000;
    Fs = 16000;
    nCh = 1;%auInfo.NumChannels;
    clipLenIn = round(clipLen_s * FsIn);
    clipLen = round(clipLen_s * Fs);

    [~,fileField] = fileparts(fileNames{j});
    fileField = ['file_' fileField];
    data.(fileField).kwClip = {};
    data.(fileField).kwRevClip = {};
    data.(fileField).backClip = {};
    data.(fileField).speechClip = {};
    data.(fileField).earlyImplantClip = {};
    data.(fileField).lateImplantClip = {};
    data.(fileField).partialEarlyClip = {};
    data.(fileField).partialLateClip = {};
    data.(fileField).shiftEarlyClip = {};
    data.(fileField).shiftLateClip = {};

    ii = find(strcmp('background', label(fileLoc)));
    background = [];
    for k = 1:length(ii)
        idx = fileLoc(ii(k));
        background = [background; audioGet(fn, [clipStart(idx) clipEnd(idx)])];
    end
    
    ii = find(strcmp('speech', label(fileLoc)));
    speech = [];
    for k = 1:length(ii)
        idx = fileLoc(ii(k));
        speech = [speech; audioGet(fn, [clipStart(idx) clipEnd(idx)])];
    end
   
    %tau = getReverbTime(fn);
    
    ii = find(strcmp('keyword', label(fileLoc)));
    for k = 1:length(ii)
        idx = fileLoc(ii(k));

        %tJitter = 0.05;
        tJitter = 0;
        jitter = round(FsIn * tJitter*(rand(1)*2-1));
        
        csIn = clipStart(idx) + jitter;
        ceIn = clipEnd(idx) + jitter;
        cdIn = ceIn - csIn;
        cd = round(cdIn * Fs/FsIn);
        
        if clipLenIn - cdIn > csIn
            disp('Discarding first keyword')
            continue
        end
        tMax = clipLen_s - tJitter;
        if cdIn > round(tMax * FsIn)
            disp('Discarding keyword: too long')
            continue
        end
        
        padIn = clipLenIn - cdIn;
        pad = round(padIn * Fs/FsIn);
        
        % keyword
        kw = audioGet(fn, [csIn-padIn+1 csIn-padIn+clipLenIn]);
        
        gain = deriveGain(kw(:,1), Fs); 
        disp(['Gain: ' num2str(gain)])
        kw = kw * gain;
        
        % window for smooth-edge implants, same one for revKw and speech
        window = getWindow(cd+1, round(0.05*Fs), size(kw,2));
        
        kwVoc = applyRandomShift(kw, Fs);
        noise = getRandomNoise(noiseDir, noiseFiles, noiseScales, clipLen, nCh);
        mix = applyRandomIR(kwVoc + noise, irDir, irFiles);
        data.(fileField).kwClip{end+1} = mix;
        
        % reversed keyword
        x = kwVoc;
        out = x(pad:pad+cd,:);
        in = flipud(x(pad:pad+cd,:));
        x(pad:pad+cd,:) = (1-window).*out + window.*in;
        noise = getRandomNoise(noiseDir, noiseFiles, noiseScales, clipLen, nCh);
        mix = applyRandomIR(x + noise, irDir, irFiles);
        data.(fileField).kwRevClip{end+1} = mix;
        
        % speech implants
        starts = 1 + floor(rand(1, alignedSpeechPerKw)*(length(speech)-clipLen));
        for n = 1:alignedSpeechPerKw
            x = kw;
            out = x(pad:pad+cd,:);
            in = gain*speech(starts(n):starts(n)+cd,:);
            x(pad:pad+cd,:) = (1-window).*out + window.*in;
            x = applyRandomShift(x, Fs);
            noise = getRandomNoise(noiseDir, noiseFiles, noiseScales, clipLen, nCh);
            mix = applyRandomIR(x + noise, irDir, irFiles);
            data.(fileField).speechClip{end+1} = mix;
        end
        starts = 1 + floor(rand(1, randomSpeechPerKw)*(length(speech)-clipLen));
        for n = 1:randomSpeechPerKw
            x = gain*speech(starts(n)+1:starts(n)+clipLen,:);
            x = applyRandomShift(x, Fs);
            noise = getRandomNoise(noiseDir, noiseFiles, noiseScales, clipLen, nCh);
            mix = applyRandomIR(x + noise, irDir, irFiles);
            data.(fileField).speechClip{end+1} = mix;
        end
        
        % partials --- 
        kwMidIn = midPt(idx);
        kwMid = round(kwMidIn * Fs/FsIn);
        
        % partial implant early
        x = kw;
        impInd = pad:pad+kwMid;
        window = getWindow(length(impInd), round(0.05*Fs), size(kw,2)); % smooth edge implants
        speechStart = 1 + floor(rand(1)*(length(speech)-length(impInd)));
        out = x(impInd,:);
        in = gain*speech(speechStart:speechStart+(length(impInd)-1),:);
        x(impInd,:) = (1-window).*out + window.*in;
        x = applyRandomShift(x, Fs);
        noise = getRandomNoise(noiseDir, noiseFiles, noiseScales, clipLen, nCh);
        mix = applyRandomIR(x + noise, irDir, irFiles);
        data.(fileField).earlyImplantClip{end+1} = mix;
        
        % partial late - missing early part
        x = kw;
        backStart = 1 + floor(rand(1)*(length(background)-length(impInd)));
        in = gain*background(backStart:backStart+(length(impInd)-1),:);
        x(impInd,:) = (1-window).*out + window.*in;
        x = applyRandomShift(x, Fs);
        noise = getRandomNoise(noiseDir, noiseFiles, noiseScales, clipLen, nCh);
        mix = applyRandomIR(x + noise, irDir, irFiles);
        data.(fileField).partialLateClip{end+1} = mix;
        
        % partial implant late
        x = kw;
        impInd = pad+kwMid+1:pad+cd;
        window = getWindow(length(impInd), round(0.05*Fs), size(kw,2)); % smooth edge implants
        speechStart = 1 + floor(rand(1)*(length(speech)-length(impInd)));
        out = x(impInd,:);
        in = gain*speech(speechStart:speechStart+(length(impInd)-1),:);
        x(impInd,:) = (1-window).*out + window.*in;
        x = applyRandomShift(x, Fs);
        noise = getRandomNoise(noiseDir, noiseFiles, noiseScales, clipLen, nCh);
        mix = applyRandomIR(x + noise, irDir, irFiles);
        data.(fileField).lateImplantClip{end+1} = mix;
        
        % partial early -  missing late part
        x = kw;
        backStart = 1 + floor(rand(1)*(length(background)-length(impInd)));
        in = gain*background(backStart:backStart+(length(impInd)-1),:);
        x(impInd,:) = (1-window).*out + window.*in;
        x = applyRandomShift(x, Fs);
        noise = getRandomNoise(noiseDir, noiseFiles, noiseScales, clipLen, nCh);
        mix = applyRandomIR(x + noise, irDir, irFiles);
        data.(fileField).partialEarlyClip{end+1} = mix;
            
        % Partials shifted to boundaries --
        % The end
        if padIn + (cdIn-kwMidIn) < csIn
            kwSh = gain*audioGet(fn, [csIn-(padIn+cdIn-kwMidIn)+1 csIn-(padIn+cdIn-kwMidIn)+clipLenIn]);
            impInd = 1:cd-kwMid;
            window = getWindow(length(impInd), round(0.05*Fs), size(kw,2));
            out = kwSh(impInd,:);
            backStart = 1 + floor(rand(1)*(length(background)-length(impInd)));
            in = gain*background(backStart:backStart+(length(impInd)-1),:);
            kwSh(impInd,:) = (1-window).*out + window.*in;
            kwSh = applyRandomShift(kwSh, Fs);
            noise = getRandomNoise(noiseDir, noiseFiles, noiseScales, clipLen, nCh);
            mix = applyRandomIR(kwSh + noise, irDir, irFiles);
            data.(fileField).shiftLateClip{end+1} = mix;
        end
        % The beginning
        if csIn+kwMidIn+clipLenIn <= auInfo.TotalSamples
            kwSh = gain*audioGet(fn, [csIn+kwMidIn+1 csIn+kwMidIn+clipLenIn]);
            impInd = clipLen-kwMid+1:clipLen;
            window = getWindow(length(impInd), round(0.05*Fs), size(kw,2));
            out = kwSh(impInd,:);
            backStart = 1 + floor(rand(1)*(length(background)-length(impInd)));
            in = gain*background(backStart:backStart+(length(impInd)-1),:);
            kwSh(impInd,:) = (1-window).*out + window.*in;
            kwSh = applyRandomShift(kwSh, Fs);
            noise = getRandomNoise(noiseDir, noiseFiles, noiseScales, clipLen, nCh);
            mix = applyRandomIR(kwSh + noise, irDir, irFiles);
            data.(fileField).shiftEarlyClip{end+1} = mix;
        end
            
    end
    
    % backgrounds
    interval = floor((length(background)-clipLen)/backPerFile);
    starts = 1:interval:interval*backPerFile;
    for n = 1:backPerFile
        noise = getRandomNoise(noiseDir, noiseFiles, noiseScales, clipLen, nCh);
        x = gain*background(starts(n):(starts(n)-1)+clipLen,:);
        x = applyRandomShift(x, Fs);
        mix = applyRandomIR(x + noise, irDir, irFiles);
        data.(fileField).backClip{end+1} = mix;
    end
    
    disp(['Saving ' fileField '...'])
    save(dataFile,'-struct','data','-append')
    clear data
    
end

function window = getWindow(winDur, rampDur, numChan)
% window function

%rampDur = round(0.05*Fs);
ramp = hanning(rampDur);
ramp = ramp(1:floor(end/2));
ramp = repmat(ramp, [1 numChan]);
window = ones(winDur, numChan);
window(1:length(ramp),:) = ramp;
window(end-length(ramp)+1:end,:) = flipud(ramp);

function gain = deriveGain(x, fs)

pRef = 0.02;
[b, a] = butter(7, [150 3400]/(fs/2));
p = sqrt(mean(filtfilt(b, a, x).^2));
gain = pRef / p;

function x = audioGet(fileName, bounds)

if ~exist('bounds','var')
    auInfo = audioinfo(fileName);
    bounds = [1 auInfo.TotalSamples];
end

[x, fs] = audioread(fileName, bounds);
x = x(:,1);
x = resample(x, 16, round(fs/1000));

function noise = getRandomNoise(noiseDir, noiseFiles, noiseScales, clipLen, nCh)

noiseIdx = randperm(length(noiseFiles), 1);
scaleIdx = randperm(length(noiseScales), 1);

noise = audioGet(fullfile(noiseDir, noiseFiles{noiseIdx}));

startIdx = randperm(length(noise)-clipLen, 1);

noise = noise(startIdx+1:startIdx+clipLen,1);
noise = noise / noiseScales(scaleIdx);

noise = repmat(noise, [1 nCh]);

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

function tau = getReverbTime(fileName)

[wav, fs] = audioread(fileName);
wav = resample(wav, 16, round(fs/1000));

[rt_est,rt_est_mean] = ML_RT_estimation(wav(:,1)', 16e3);
    
tau = rt_est_mean;

function xOut = applyRandomIR(x, irDir, irFiles)

orMix = 1.0;
irMix = 1.0;

% randomly don't apply any IR, but do some random gain
if randperm(length(irFiles)+1,1)==1
    xOut = x * (1+rand(1)*0.5);
    return
end

irIdx = randperm(length(irFiles), 1);

IR = audioGet(fullfile(irDir, irFiles{irIdx}));

xOut = orMix*x + irMix*applyIR(x, IR);
