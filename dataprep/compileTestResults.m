function [decMat, probMat, idMatch, condMatch] = compileTestResults(probFile, feaFile, th, ep)

load(probFile) % -> prob

if ndims(prob) == 3 
    if exist('ep','var')
        prob = squeeze(prob(ep,:,2:end));
    else
        nClass = size(prob,3);
        for j = 2:nClass
            score(:,:,j-1) = (prob(:,:,j) > 0.2) & (prob(:,:,j) > max(prob(:,:,setdiff(2:nClass,j)),[],3));
        end
        prob = squeeze(sum(score,2));
    end
else
    prob = prob(:,2:end);
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

for l = 1:size(prob, 2)
    p = prob(:,l);
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
