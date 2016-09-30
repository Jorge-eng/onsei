
% first step: batchrunner of a particular model
% on a particular condition.
%
% As a result, there will be dirNames for 
% several conditions. Some negative, some positive.


modelName = 'model_aug30_lstm_med_dist_okay_sense_stop_snooze_tiny_912_ep216';

% The number of posDirs will be the number of keywords 
% in the model outputs
posDirs = {'testingWavs_kwClip_okay_sense',...
           'testingWavs_kwClip_stop',...
           'testingWavs_kwClip_snooze'}; 
% The number of negDirs is arbitrary
negDirs = {'noiseDataset/16k',...
           'reverberant_speech/16k'};
%negDirs = {'speechDataset/16k'};

% Define the threshold sweeep.
nTh = 5;
ths = linspace(0.5, 0.95, 7) * 4095;
counts = 1:20;

clear posRate
for j = 1:length(posDirs)
    dirName = fullfile('../net/outputs',modelName,posDirs{j})
    [num, tot] = runnerCsvToROC(dirName, ths, counts);
    posRate(:,:,j) = squeeze(sum(num(:,:,:,j))) / size(num,1);
end
clear faRate faNum
for j = 1:length(negDirs)
    dirName = fullfile('../net/outputs',modelName,negDirs{j})
    [num, tot] = runnerCsvToROC(dirName, ths, counts);
    for k = 1:size(num,4)
        faRate(:,:,k,j) = squeeze(sum(num(:,:,:,k))) / (sum(tot)*0.015/60/60);
        faNum(:,:,k,j) = squeeze(sum(num(:,:,:,k)));
    end
end

%%
figure
subplot(131)
plot(faRate(1:2:20,:,1,1), posRate(1:2:20,:,1,1), '*-')
set(gca,'xlim',[0 1],'ylim',[0 1]), grid on
title('OKAY_SENSE')
legend(strsplit(num2str(ths/4095)),'location','southeast')
ylabel('Detection rate')
xlabel('False alarms / hour')

subplot(132)
plot(faRate(1:2:20,:,2,1), posRate(1:2:20,:,2,1), '*-')
set(gca,'xlim',[0 1],'ylim',[0 1]), grid on
title('STOP')
legend(strsplit(num2str(ths/4095)),'location','southeast')
ylabel('Detection rate')
xlabel('False alarms / hour')

subplot(133)
plot(faRate(1:2:20,:,3,1), posRate(1:2:20,:,3,1), '*-')
set(gca,'xlim',[0 1],'ylim',[0 1]), grid on
title('SNOOZE')
legend(strsplit(num2str(ths/4095)),'location','southeast')
ylabel('Detection rate')
xlabel('False alarms / hour')
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
