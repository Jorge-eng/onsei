
probFile = '../net/prob_all_lstm_small_dist_okay_sense_tiny_819_dist3.mat';
testFile = 'testingFeats_okay_sense_tiny.mat';
data = [];
for ep = 51:300
    [mat, data] = compileTestResults(probFile, testFile, 0.2, ep, data);
    for j = 1:size(mat,3), subplot(size(mat,3),1,j), imagesc(mat(:,:,j),[0 1]), end
    ep, pause
end
%%
clear cd1 cd2 fa
probFile = '../net/prob_all_lstm_small_okay_sense_tiny_819_15.mat';
testFile = 'testingFeats_okay_sense_tiny.mat';
decTh = 0.5;
data = [];
for ep = 1:300, ep
    [mat,data] = compileTestResults(probFile,testFile, decTh, ep, data);
    mat(isnan(mat)) = 0;
    cd1(ep) = 0;
    cd2(ep) = 0;
    fa(ep) = 0;
    for j = 1:size(mat,3)
        cd1(ep) = cd1(ep) + sum(mat(:,j,j)>0.7);
        cd2(ep) = cd2(ep) + sum(mat(:,j,j));
        fa(ep) = sum(sum(mat(:,setdiff(1:end, j),j)));
    end
end
%%
clear mats
for ep = 1:length(ii)
    %figure
    [mat,data] = compileTestResults(probFile, testFile, decTh, ii(ep), data);
    mats(:,:,:,ep) = mat;
    for j = 1:size(mat,3), subplot(size(mat,3),1,j), imagesc(mat(:,:,j),[0 1]), end
    ii(ep), cd2(ii(ep)), fa(ii(ep)), pause
end
