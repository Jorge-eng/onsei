th = 0.5;

[mat, p] = compileTestResults('../net/prob_cleanData_cleanModel.mat','testingFeats.mat', th);
subplot(2, 4, 1)
imagesc(mat, [0 1])

[mat, p] = compileTestResults('../net/prob_noiseData_cleanModel.mat','testingFeatsNoise.mat', th);
subplot(2, 4, 2)
imagesc(mat, [0 1])

[mat, p] = compileTestResults('../net/prob_cleanData_noiseModel.mat','testingFeats.mat', th);
subplot(2, 4, 3)
imagesc(mat, [0 1])

[mat, p] = compileTestResults('../net/prob_noiseData_noiseModel.mat','testingFeatsNoise.mat', th);
subplot(2, 4, 4)
imagesc(mat, [0 1])

[mat, p] = compileTestResults('../net/prob_cleanData_cleanModel_tiny.mat','testingTinyfeats.mat', th);
subplot(2, 4, 5)
imagesc(mat, [0 1])

[mat, p] = compileTestResults('../net/prob_noiseData_cleanModel_tiny.mat','testingTinyfeatsNoise.mat', th);
subplot(2, 4, 6)
imagesc(mat, [0 1])

[mat, p] = compileTestResults('../net/prob_cleanData_noiseModel_tiny.mat','testingTinyfeats.mat', th);
subplot(2, 4, 7)
imagesc(mat, [0 1])

[mat, p] = compileTestResults('../net/prob_noiseData_noiseModel_tiny.mat','testingTinyfeatsNoise.mat', th);
subplot(2, 4, 8)
imagesc(mat, [0 1])
 