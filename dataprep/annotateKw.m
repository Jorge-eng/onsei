function midPt = annotateKw(dirName, csvFile)

%dirName = '/Users/hello/Dropbox/Data/keyword';
%csvFile = '/Users/hello/Dropbox/Data/keyword/annotations.csv';

fileName = {};
clipStart = [];
clipEnd = [];
label = {};

fid = fopen(csvFile,'r');
while 1
    line = fgetl(fid);
    if ~ischar(line)
        break
    end
    
    line = strsplit(line,',');
    
    fileName{end+1} = line{1};
    clipStart(end+1) = str2double(line{2});
    clipEnd(end+1) = str2double(line{3});
    label{end+1} = line{4};
end
fclose(fid);

ii = find(strcmp('keyword', label));

Fs = 48000;
clipLen = 2 * Fs;

windowLen = Fs * 0.032;
WINDOW = hann(windowLen);
NFFT = 2^nextpow2(windowLen);
NOVERLAP = windowLen / 2;
frameRate = Fs / NOVERLAP;

midPt = {};
for j = 1:length(ii)
    idx = ii(j);
    
    fn = fullfile(dirName, fileName{idx});
    
    cs = clipStart(idx);
    ce = clipEnd(idx);
    cd = ce - cs;
    
    pad = floor((clipLen - cd) / 2);
    
    wav = audioread(fn, [cs-pad+1 cs-pad+clipLen]);
    
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
    soundsc(wav(wavPts+1:end,:), Fs)
   
    midPt{end+1} = wavPts;
    
end
