if 0
kw = 'snooze';
iter = '22';
inTag = 'iter';
outTag = 'iter2';
winLen_s = 1.6;
load(['../net/prob_long_' kw '_' inTag '_' iter '.mat'])
outDir = ['~/Dropbox/Data/keyword/falseAlarms_' kw '_' iter];
[~,~] = mkdir(outDir);
th = 0.5;
ii = find(prob(:,2) > th);
p = prob(ii,2);

fs = 16e3;
frameShift = 0.01 * fs;
winLen = winLen_s * fs;

startSamps = 1 + frameShift*startTimes(ii);
endSamps = startSamps + winLen - 1;

files = cellstr(files);
for j = 1:length(ii), j
    idx = ii(j);
    file = files{1+fileIdx(idx)};
    clip = audioread(file, double([startSamps(j) endSamps(j)]));
    
    outName = ['speechClip_' kw '_fa_' outTag '_' num2str(j) '.wav'];
    audiowrite(fullfile(outDir, outName), clip, fs)
    %if file ~= lastfile
    %    wav = audioread(file);
    %end
    %lastfile = file;
    
end
end
%%
modelName = 'model_aug30_lstm_med_dist_okay_sense+stop+snooze_tiny_fa_1006_ep077';
%kws = {'okay_sense','stop','snooze'};
kws = {'okay_sense'};
negDirs = {'fa_pool'};
file = {};
det = [];
fileType = 'mat';
c = 1; 
[~,~] = mkdir('~/keyword/fa_pool/clips');
for k = 1:length(negDirs)
    dirName = fullfile('../net/outputs',modelName,negDirs{k})

    d = dir(fullfile(dirName, ['*.' fileType]));
    h = ones(3, 1, 'single');
    th_pct = 0.5;
    
    for j = 1:length(d)
        disp(d(j).name)
        if strcmp(fileType, 'csv')
            data = csvread(fullfile(dirName, d(j).name));
            th = th_pct * 4095;
        elseif strcmp(fileType, 'mat')
            data = load(fullfile(dirName, d(j).name));
            data = squeeze(data.prob);
            th = th_pct;
        end
        
        [num, loc] = runnerDetector(data(:,2:end), h, th);
        for kw = 1:length(kws)
            for l = 1:num(kw)
                file{end+1} = fullfile(dirName, d(j).name);
                det(end+1) = loc{kw}(l);
                
                if 1
                    wavFile = strrep(file{end},fullfile('../net/outputs',modelName),'~/keyword');
                    if strcmp(fileType, 'csv')
                        wavFile = strrep(wavFile,'.csv','');
                    elseif strcmp(fileType, 'mat')
                        wavFile = strrep(wavFile,'.bin.mat','');
                    end
                    
                    lim = [1 1.6*16e3] + round(16e3*(det(end)*0.015-1.5));
                    if lim(1) < 1
                        lim = lim - lim(1) + 1;
                    end
                    wav = audioread(wavFile,lim);
                    outFile = ['~/keyword/fa_pool/clips/falseAlarmClip_' kws{kw} '_' num2str(c) '.wav'];
                    disp(outFile)
                    audiowrite(outFile,wav,16e3);
                    c = c + 1;
                end
            end
        end
        
    end
end
