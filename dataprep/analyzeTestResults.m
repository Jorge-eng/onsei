
probFile = '../net/prob_all_small_sigm_okay_sense+alexa_tiny.mat';
testFile = 'testingFeats_okay_sense+alexa_tiny.mat';
for ep = 1:200
    [mat,p] = compileTestResults(probFile, testFile, 0.1, ep);
    for j = 1:size(mat,3), subplot(size(mat,3),1,j), imagesc(mat(:,:,j),[0 1]), end
    ep, pause
end
%%
clear cd1 cd2 fa
probFile = '../net/prob_all_24x24_okay_sense_tiny_816.mat';
testFile = 'testingFeats_okay_sense_tiny.mat';
decTh = 0.5;%1e-6;
for ep = 1:300, ep
    [mat,p] = compileTestResults(probFile,testFile, decTh, ep);
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
    [mat,p] = compileTestResults(probFile, testFile, decTh, ii(ep));
    mats(:,:,:,ep) = mat;
    for j = 1:size(mat,3), subplot(size(mat,3),1,j), imagesc(mat(:,:,j),[0 1]), end
    ii(ep), cd2(ii(ep)), fa(ii(ep)), pause
end
