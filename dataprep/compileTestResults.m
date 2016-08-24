function [decMat, data] = compileTestResults(probFile, feaFile, th, ep, data)

if ~exist('data','var') || isempty(data)
    load(probFile) % -> prob
    data = prob;
else
    prob = data;
end
%prob = prob(:,:,:,1:2);

nd = ndims(prob);
if nd == 4 || (nd == 3 && ~exist('ep','var'))
    if nd == 4
        prob = squeeze(max(prob(:,:,end-20:end,:),[],3));
    elseif nd == 3
        prob = squeeze(max(prob(:,end-20:end,:),[],2));
    end
end
if exist('ep','var')
    prob = reshape(prob(ep,:,:),[size(prob,2) size(prob,3)]);
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
            if 0
                dec = dec & isMax(ii);
            end
            posRate = sum(dec) / length(dec);
            
            decMat(j, k, l) = posRate;
            probMat(j, k, l) = mean(p(ii));
            
        end
    end
end
