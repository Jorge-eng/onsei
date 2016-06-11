function [decMat, probMat, idMatch, condMatch] = compileTestResults(probFile, feaFile, th)

load(probFile) % -> prob
p = prob(:,2);

load(feaFile); % -> idMatch, condMatch, identity, sampleType

idMatch = cellstr(idMatch);
condMatch = cellstr(condMatch);
nId = length(idMatch);
nCond = length(condMatch);

id = cellstr(identity);
cond = cellstr(sampleType);

decMat = zeros(nId, nCond);
probMat = zeros(nId, nCond);

for j = 1:nId
    for k = 1:nCond
        
        ii = strcmp(idMatch{j}, id) & strcmp(condMatch{k}, cond);
        dec = p(ii) > th;
        posRate = sum(dec) / length(dec);
        
        decMat(j, k) = posRate;
        probMat(j, k) = mean(p(ii));
        
    end
end
