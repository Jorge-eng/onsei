function [num, loc] = runnerDetector(data, h, th)
% [num, loc] = runnerDetector(data, h, th)

if length(h) == 1
    h = ones(h, 1, 'single');
end

for kw = 1:size(data, 2)
    det = single(data(:,kw) > th);
    det = conv2(h, 1, det, 'same');
    det = single(det==length(h));
    det = diff(det) > 0;
    loc{kw} = find(det);
    num(kw) = length(find(det));
end
