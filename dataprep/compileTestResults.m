function [decMat, probMat, idMatch, condMatch] = compileTestResults(probFile, feaFile, th, ep)

load(probFile) % -> prob
if ndims(prob) == 3
    prob = squeeze(prob(ep,:,:));
end
load(feaFile); % -> idMatch, condMatch, identity, sampleType

idMatch = cellstr(idMatch);
condMatch = cellstr(condMatch);
nId = length(idMatch);
nCond = length(condMatch);

id = cellstr(identity);
cond = cellstr(sampleType);

decMat = zeros(nId, nCond);
probMat = zeros(nId, nCond);

for l = 1:size(prob, 2)-1
    p = prob(:,l+1);
    isMax = (p == max(prob,[],2));
    for j = 1:nId
        for k = 1:nCond
            
            ii = strcmp(idMatch{j}, id) & strcmp(condMatch{k}, cond);
            dec = p(ii) > th;
            if 1
                dec = dec & isMax(ii);
            end
            posRate = sum(dec) / length(dec);
            
            decMat(j, k, l) = posRate;
            probMat(j, k, l) = mean(p(ii));
            
        end
    end
end
