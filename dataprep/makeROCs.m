
% first step: batchrunner of a particular model
% on a particular condition.
%
% As a result, there will be dirNames for 
% several conditions. Some negative, some positive.

modelName = 'model_aug30_lstm_med_dist_okay_sense_tiny_95_1009_ep150';

% The number of posDirs will be the number of keywords 
% in the model outputs
posDirs = {'testingWavs_kwClip_okay_sense'};
%posDirs = {'testingWavs_kwClip_okay_sense',...
%           'testingWavs_kwClip_stop',...
%           'testingWavs_kwClip_snooze'}; 
% The number of negDirs is arbitrary
%negDirs = {'noiseDataset/16k',...
%           'reverberant_speech/16k'};%,...
           %'testingWavs_adversarial'};
%negDirs = {'speechDataset/16k'};
negDirs = {'noiseDataset/16k',...
           'reverberant_speech/16k'};%,...
%negDirs = {'TED'};       
% Define the threshold sweeep.
nTh = 14;
ths = linspace(0.05, 0.95, nTh);
if 0
    ths = ths * 4095;
    fileType = 'csv';
else
    fileType = 'mat';
end
%ths = combvec(ths, ths);
counts = 1:20;
%%
clear posRate
for j = 1:length(posDirs)
    dirName = fullfile('../net/outputs',modelName,posDirs{j})
    [num, tot] = runnerCsvToROC(dirName, ths, counts, fileType);
    posRate(:,:,j) = squeeze(sum(num(:,:,:,j))) / size(num,1);
end
%%
clear faRate faNum
for j = 1:length(negDirs)
    dirName = fullfile('../net/outputs',modelName,negDirs{j})
    [num, tot] = runnerCsvToROC(dirName, ths, counts, fileType);
    for k = 1:size(num,4)
        faRate(:,:,k,j) = squeeze(sum(num(:,:,:,k))) / (sum(tot)*0.015/60/60);
        faNum(:,:,k,j) = squeeze(sum(num(:,:,:,k)));
    end
end

%%
xLim = [1 7];
for pl = 1:size(faRate, 4)
    for kw = 1:length(posDirs)
        figure
        set(gcf,'Name',modelName,'WindowStyle','docked')
        subplot(1,length(posDirs),1)
        plot(faRate(1:2:20,:,kw,pl), posRate(1:2:20,:,1,1), '*-')
        set(gca,'xlim',[0 xLim(pl)],'ylim',[0 1]), grid on
        title('OKAY SENSE')
        %legend(strsplit(num2str(ths/4095)),'location','southeast')
        ylabel('Detection rate')
        xlabel('False alarms / hour')
    end
end
%%
figure
subplot(131)
pl = 2;
plot(faNum(1:2:20,:,1,pl)/size(num,1)*100, posRate(1:2:20,:,1), '*-')
set(gca,'xlim',[0 .7],'ylim',[0 1]), grid on
title('OKAY SENSE')
legend(strsplit(num2str(ths/4095)),'location','southeast')
ylabel('Detection rate')
xlabel('False alarms / 100 utterances')

subplot(132)
plot(faNum(1:2:20,:,2,pl)/size(num,1)*100, posRate(1:2:20,:,2), '*-')
set(gca,'xlim',[0 .7],'ylim',[0 1]), grid on
title('STOP')
legend(strsplit(num2str(ths/4095)),'location','southeast')
ylabel('Detection rate')
xlabel('False alarms / 100 utterances')

subplot(133)
plot(faNum(1:2:20,:,3,pl)/size(num,1)*100, posRate(1:2:20,:,3), '*-')
set(gca,'xlim',[0 .7],'ylim',[0 1]), grid on
title('SNOOZE')
legend(strsplit(num2str(ths/4095)),'location','southeast')
ylabel('Detection rate')
xlabel('False alarms / 100 utterances')

%%

for k = 1:size(posRate,3)
    
    imin = find(posRate(:,end,k) >= 0.7, 1, 'last');
    faMin(k) = min(faRate(imin,end,k,3));
    
end
%%
modelName = 'model_aug30_lstm_med_dist_TRAIN_40_prev_1008_ep254';
kws = {'okay_sense','stop','snooze'};
file = {};
det = [];
c = 5000; 
[~,~] = mkdir('~/keyword/new_fa');
for k = 1:length(negDirs)
    dirName = fullfile('../net/outputs',modelName,negDirs{k})

    d = dir(fullfile(dirName, '*.csv'));
    h = ones(5, 1, 'single');
    th = 0.1 * 4095;
    
    for j = 1:length(d)
        disp(d(j).name)
        data = csvread(fullfile(dirName, d(j).name));
        
        [num, loc] = runnerDetector(data(:,2:end), h, th);
        for kw = 1:length(num)
            for l = 1:num(kw)
                file{end+1} = fullfile(dirName, d(j).name);
                det(end+1) = loc{kw}(l);
                
                if 1
                    wavFile = strrep(file{end},fullfile('../net/outputs',modelName),'~/keyword');
                    wavFile = strrep(wavFile,'.csv','');
                    
                    lim = [1 1.6*16e3] + round(16e3*(det(end)*0.015-1.5));
                    if lim(1) < 1
                        lim = lim - lim(1) + 1;
                    end
                    wav = audioread(wavFile,lim);
                    outFile = ['~/keyword/new_fa/speechRandomClip_' kws{kw} '_fa_' num2str(c) '.wav'];
                    disp(outFile)
                    audiowrite(outFile,wav,16e3);
                    c = c + 1;
                end
            end
        end
        
    end
end
%%
clear wav
for j = 1:length(file), j
    wavFile = strrep(file{j},fullfile('../net/outputs',modelName),'~/keyword');
    wavFile = strrep(wavFile,'.csv','');
    lim = [1 1.6*16e3] + round(16e3*(det(j)*0.015-1.5));
    if lim(1) < 1
        lim = lim - lim(1) + 1;
    end
    wav{j} = audioread(wavFile,lim);
end
