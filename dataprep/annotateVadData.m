function annotateVadData(dataFile)
% annotateVadData(dataFile)

%{
Algorithm:
- endpoint each file 
- use negative labels just prior and after pad
- use positive labels just after and prior to interior
- use positive labels randomly in interior
- use negative labels randomly in clips with no speech mixed in
%}

if ~exist('dataFile','var')
    dataFile = 'clipData.mat';
end

Fs = 16000;
clipLen_s = 4.9;
maxSpeechLen_s = 3.5;
clipLen = round(clipLen_s * Fs);
backPerFile = 1;
nCh = 1;
    
speechDir = '~/keyword/reverberant_speech/16k';
d = dir(fullfile(speechDir, '*.wav'));
speechFiles = {d.name};
dur = zeros(size(speechFiles));
for j = 1:length(d)
    info = audioinfo(fullfile(speechDir, speechFiles{j}));
    dur(j) = info.Duration;
end
speechFiles = speechFiles(dur <= maxSpeechLen_s);
speechFiles = speechFiles(randperm(length(speechFiles), 1000));

noiseDir = '~/Dropbox/Data/keyword/noiseWavs';
d = dir(fullfile(noiseDir, '*.wav'));
noiseFiles = {d.name};
noiseScales = 1./10.^((-18:-2:-30)/20);

irDir = 'IR';
d = dir(fullfile(irDir,'*16k.wav'));
irFiles = {d.name};

fileField = ['file_all'];
annoField = ['annotation_all'];
vars = {'backClip','speechRandomClip'};
for v = 1:length(vars)
    data.(fileField).(vars{v}) = {};
    data.(annoField).(vars{v}) = {};
end

for j = 1:length(speechFiles)
    disp(['File ' num2str(j) ' out of ' num2str(length(speechFiles))])
    
    fn = fullfile(speechDir, speechFiles{j});
    
    [speech, fs] = audioread(fn);
    
    gain = deriveGain(speech, Fs);
    disp(['Gain: ' num2str(gain)])
    speech = speech * gain;
    
    wav = zeros(clipLen, 1);
    pad = round((length(wav) - length(speech))/2);
    wav(pad+1:pad+length(speech)) = speech;
    [ptExt, ptInt] = endpointer(wav);
    
    noise = getRandomNoise(noiseDir, noiseFiles, noiseScales, clipLen, nCh);
    mix = applyRandomIR(wav + noise, irDir, irFiles);
    
    data.(fileField).speechRandomClip{end+1} = mix;
    
    trDur = 0.03*Fs;
    beforeStart = [ptExt(1)-trDur ptExt(1)]/clipLen;
    afterStart = [ptInt(1) ptInt(1)+trDur]/clipLen;
    beforeEnd = [ptInt(2)-trDur ptInt(2)]/clipLen;
    afterEnd = [ptExt(2) ptExt(2)+trDur]/clipLen;
    interior = (ptInt(1) + rand(1,3)*(ptInt(2)-ptInt(1)-trDur*2))/clipLen;
    interiorData = {};
    for k = 1:length(interior)
        interiorData{end+1} = {'speech',[interior(k) interior(k)]};
    end
    data.(annoField).speechRandomClip{end+1} = {{'back',beforeStart},...
                                                {'speech',afterStart},...
                                                {'speech',beforeEnd},...
                                                {'back',afterEnd},...
                                                interiorData{:}};
                                            
    % backgrounds
    for n = 1:backPerFile
   
        x = zeros(clipLen, 1);
        noise = getRandomNoise(noiseDir, noiseFiles, noiseScales, clipLen, nCh);
    
        mix = applyRandomIR(x + noise, irDir, irFiles);
        data.(fileField).backClip{end+1} = mix;
        
        interior = (0.2*Fs + rand(5)*(clipLen-0.2*Fs))/clipLen;
        interiorData = {};
        for k = 1:length(interior)
            interiorData{end+1} = {'back',[interior(k) interior(k)]};
        end
        
        data.(annoField).backClip{end+1} = {interiorData{:}};
    end
    
end

disp(['Saving ' fileField '...'])
save(dataFile,'-struct','data','-v7.3')

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
