
% first step: batchrunner of a particular model
% on a particular condition.
%
% As a result, there will be dirNames for 
% several conditions. Some negative, some positive.

%modelName = 'model_aug30_lstm_med_dist_okay_sense_stop_snooze_tiny_fa_1006_ep077';
%modelName = 'model_aug30_lstm_med_dist_okay_sense_stop_snooze_tiny_912_ep216';
%modelName = 'model_aug30_lstm_med_dist_okay_sense+stop+snooze_tiny_912_ep216';
modelName = 'model_aug30_lstm_med_dist_okay_sense+stop+snooze_tiny_fa8_1014_ep130';
%modelName = 'model_aug30_lstm_med_dist_okay_sense_stop_snooze_tiny_fa8_1014_ep105';

% The number of posDirs will be the number of keywords 
% in the model outputs
%posDirs = {'testingWavs_kwClip_okay_sense'};
posDirs = {'testingWavs_kwClip_okay_sense',...
           'testingWavs_kwClip_stop',...
           'testingWavs_kwClip_snooze'}; 
% The number of negDirs is arbitrary
%negDirs = {'noiseDataset/16k',...
%           'reverberant_speech/16k'};%,...
           %'testingWavs_adversarial'};
%negDirs = {'fa_pool'};
negDirs = {'noiseDataset/16k',...
           'reverberant_speech/16k',...};%,...%};%,...
           'TED'};
%negDirs = {'TED'};       
% Define the threshold sweeep.
%nTh = 15;
%ths = linspace(0.05, 1.00, nTh);
%ths = [0.05 0.3 0.5 0.6 0.7 0.8 0.9 0.95 0.99 1.00];
ths_pct = [0.2 0.5 0.6 0.7 0.8 0.9 0.95 0.98];
if 0
    ths = ths_pct * 4095;
    fileType = 'csv';
else
    ths = ths_pct;
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
kws = {'OKAY SENSE','STOP','SNOOZE'};
xLim = [1 3 3];
for pl = 1:size(faRate, 4)
    figure
    %set(gcf,'Name',modelName,'WindowStyle','docked')
    for kw = 1:length(posDirs)
        subplot(1,length(posDirs),kw)
        plot(faRate(1:2:20,:,kw,pl), posRate(1:2:20,:,kw,1), '*-')
        set(gca,'xlim',[0 xLim(pl)],'ylim',[0 1]), grid on
        title(kws{kw})
        legend(strsplit(num2str(ths_pct)),'location','southeast')
        ylabel('Detection rate')
        xlabel('False alarms / hour')
    end
end
%{
%%

for k = 1:size(posRate,3)
    
    imin = find(posRate(:,end,k) >= 0.7, 1, 'last');
    faMin(k) = min(faRate(imin,end,k,3));
    
end
%%
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
%}