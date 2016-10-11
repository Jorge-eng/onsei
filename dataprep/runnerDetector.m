function [num, loc] = runnerDetector(data, h, th)
% [num, loc] = runnerDetector(data, h, th)

if length(h) == 1
    h = ones(h, 1, 'single');
end
if length(th)==1
    th = [th th];
end

for kw = 1:size(data,2)/2
    if 1
        det = single(data(:,kw) > th(1));
        det = conv2(h, 1, det, 'same');
        det = single(det==length(h));
        det = diff(det) > 0;
        
        if 0
            gate = single(data(:,kw+3) > th(2));
            gate = conv2(h, 1, gate, 'same');
            gate = single(gate==length(h));
            gate = single(diff(gate) > 0);
            g = zeros(101,1,'single');
            g(51:end) = 1;
            gate = conv2(g, 1, gate, 'same');
            
            det = det .* gate;
        end
    else
        
        g = zeros(101,1,'single');
        g(51:end) = 1;
        %det = 1/2*exp(conv2(g/sum(g),1,log(1+data(:,kw+3)),'same')+log(1+data(:,kw)));
        det = exp(1/2*(log(1+conv2(g/sum(g),1,data(:,kw+3),'same'))+log(1+data(:,kw))));
        det = single(det > th);
        det = conv2(h, 1, det, 'same');
        det = single(det==length(h));
        det = diff(det) > 0;
    end
    
    loc{kw} = find(det);
    num(kw) = length(find(det));
end
