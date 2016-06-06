function midPt = annotateKw(dirName, csvFile)

%dirName = '/Users/hello/Dropbox/Data/keyword';
%csvFile = '/Users/hello/Dropbox/Data/keyword/annotations.csv';

fileName = {};
clipStart = [];
clipEnd = [];
label = {};
rawline = {};

fid = fopen(csvFile,'r');
while 1
    line = fgetl(fid);
    if ~ischar(line)
        break
    end
    
    splitline = strsplit(line,',');
    
    fileName{end+1} = splitline{1};
    clipStart(end+1) = str2double(splitline{2});
    clipEnd(end+1) = str2double(splitline{3});
    label{end+1} = splitline{4};
    
    % discard existing extra annotations, if any
    rawline{end+1} = strjoin(splitline(1:4),',');
end
fclose(fid);

Fs = 48000;
windowLen = Fs * 0.032;
WINDOW = hann(windowLen);
NFFT = 2^nextpow2(windowLen);
NOVERLAP = windowLen / 2;
frameRate = Fs / NOVERLAP;

fid = fopen('temp.csv', 'w');

% Write out further annotated keywords
ii = find(strcmp('keyword', label));
midPt = {};
for j = 1:length(ii)
    idx = ii(j);
    
    fn = fullfile(dirName, fileName{idx});
    
    cs = clipStart(idx);
    ce = clipEnd(idx);
    
    wav = audioread(fn, [cs+1 ce]);
    
    S = spectrogram(wav(:,2),WINDOW,NOVERLAP,NFFT,Fs);

    clf
    axes('position',[0.1300    0.2872    0.7750    0.6378])
    imagesc(abs(S).^(1/3)); axis xy
    axes('position',[ 0.1300    0.1100    0.7750    0.1243])
    plot(wav); axis tight

    disp(['Select Keyword Midpoint (' num2str(j) ')'])
    pts = ginput(1);
    pts = pts(:,1) / frameRate;
    wavPts = round(pts * Fs);
    
    soundsc(wav(1:wavPts,:), Fs)
    pause(wavPts/Fs)
    soundsc(wav(wavPts+1:end,:), Fs)
   
    midPt{end+1} = wavPts;
    
    fprintf(fid, '%s\n', [rawline{idx} ',' num2str(wavPts)]);
end

% Write out everything else as it was
ii = setdiff(1:length(label), ii);
for j = 1:length(ii)
    fprintf(fid, '%s\n', rawline{ii(j)});
end
fclose(fid);

movefile('temp.csv', csvFile)
