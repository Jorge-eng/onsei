
probFile = '../net/prob_all_model_aug30_lstm_med_dist_TRAIN_00_1007.mat';
testFile = 'testingFeats_okay_sense+stop+snooze_tiny2.mat';
data = [];
for ep = 1:300
    [mat, num, tot, data] = compileTestResults(probFile, testFile, 0.2, ep, data);
    for j = 1:size(mat,3), subplot(size(mat,3),1,j), imagesc(mat(:,:,j),[0 1]), end
    ep, pause
end
%%
clear cd1 cd2 fa cm
probFile = '../net/prob_all_model_aug30_lstm_med_dist_okay_sense+stop+snooze_tiny_fa8_1014.mat';
testFile = 'testingFeats_okay_sense+stop+snooze_tiny_fa8.mat';
decTh = 0.50;
data = [];
figure
for ep = 1:300, ep
    [mat,num,tot,data] = compileTestResults(probFile,testFile, decTh, ep, data);
    for j = 1:size(mat,3), subplot(size(mat,3),1,j), imagesc(mat(:,:,j),[0 1]), end, drawnow
    mat(isnan(mat)) = 0;
    cd1(ep) = 0;
    cd2(ep) = 0;
    fa(ep) = 0;
    for j = 1:size(mat,3)
        %if j == 2, continue; end        
        cd1(ep) = cd1(ep) + sum(mat(:,j,j)>0.7);
        cd2(ep) = cd2(ep) + sum(mat(:,j,j));
        fa(ep) = fa(ep) + sum(sum(mat(:,setdiff(1:end, j),j)));
    end
end
%%
clear mats cm nn
data = [];
for ep = 1:length(ii)
    %figure
    [mat,num,tot,data] = compileTestResults(probFile, testFile, decTh, ii(ep), data);
    mats(:,:,:,ep) = mat;
    figure, set(gcf,'WindowStyle','docked'), set(gcf,'Name',['ep' num2str(ii(ep))])
    for j = 1:size(mat,3), subplot(size(mat,3),1,j), imagesc(mat(:,:,j),[0 1]), end
    ii(ep),% cd2(ii(ep)), fa(ii(ep)), pause
    
        %cm = zeros(size(mat,3),size(mat,3)+1,ep);
    for j = 1:size(mat,3)
        for k = 1:size(mat,3)
            cm(j,k,ep) = sum(num(:,k,j)) / sum(tot(:,k,j));
            nn(j,k,ep) = sum(tot(:,k,j));
        end
        cm(j,size(mat,3)+1,ep) = sum(sum(num(:,k+1:end,j))) / sum(sum(tot(:,k+1:end,j)));
        nn(j,size(mat,3)+1,ep) = sum(sum(tot(:,k+1:end,j)));
    end
end
% using max: 157, 277, 298
% using mean: 150, 259, 277, 289, 298, 300

%Large:
% 104, 105
% 70, 77, 96, 105, 266

% TOP 3: 105, 96, 266

%Med:
% 175, 217
% 96, 125, 272

% 40_1007
% 57(56) 161(160)

% 40_pre_1008
% 43(42)* 45(44) 82(81) 133(132)* 191(190) 255(254)

% 95_pre_1008
% 57(56)* 101(100) 103(102)*

% med: 91(90) , 259(258)
% 24x24: 207(206), 261(260), 274(273)