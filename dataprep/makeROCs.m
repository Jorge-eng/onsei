
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
negDirs = {'noiseDataset/16k'};

% Define the threshold sweeep.
nTh = 20;
ths = linspace(4095/nTh,4095,nTh);
counts = 1:20;

for j = 1:length(posDirs)
    dirName = fullfile('../net/outputs',modelName,posDirs{j})
    [num, tot] = runnerCsvToROC(dirName, ths, counts);
    posRate(:,:,j) = squeeze(sum(num(:,:,:,j))) / size(num,1);
end
for j = 1:length(negDirs)
    dirName = fullfile('../net/outputs',modelName,negDirs{j})
    [num, tot] = runnerCsvToROC(dirName, ths, counts);
    for k = 1:size(num,4)
        faRate(:,:,k,j) = squeeze(sum(num(:,:,:,k))) / (sum(tot)*0.015/60/60);
    end
end

