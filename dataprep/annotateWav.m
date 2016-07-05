function [a, label] = annotateWav(wavFile, csvFile)

%wavFile = '/Users/hello/Dropbox/160517_02.WAV';
[wav, Fs] = audioread(wavFile);

maxLen_s = 120;
if length(wav)/Fs > maxLen_s
    wav = wav(1:maxLen_s*Fs,:);
end

windowLen = round(Fs * 0.032);
WINDOW = hann(windowLen);
NFFT = 2^nextpow2(windowLen);
NOVERLAP = round(windowLen / 2);
frameRate = Fs / NOVERLAP;

S = spectrogram(wav(:,1),WINDOW,NOVERLAP,NFFT,Fs);

clf
axes('position',[0.1300    0.2872    0.7750    0.6378])
imagesc(abs(S).^(1/3)); axis xy
axes('position',[ 0.1300    0.1100    0.7750    0.1243])
plot(wav); axis tight

disp('Keywords:')
N = input('How many?');
a = {};
label = {};
for j = 1:N
    
   disp(['Select Points (' num2str(j) ')'])
   pts = ginput(2);
   pts = pts(:,1) / frameRate;
   wavPts = round(pts * Fs);
   
   soundsc(wav(wavPts(1):wavPts(2),:), Fs)
   
   a{end+1} = wavPts;
   label{end+1} = 'keyword';
   
end

disp('Speech:')
N = input('How many?');
for j = 1:N
    
   disp(['Select Points (' num2str(j) ')'])
   pts = ginput(2);
   pts = pts(:,1) / frameRate;
   wavPts = round(pts * Fs);
   
   soundsc(wav(wavPts(1):wavPts(2),:), Fs)
   
   a{end+1} = wavPts;
   label{end+1} = 'speech';
   
end

disp('Background:')
N = input('How many?');
for j = 1:N
    
   disp(['Select Points (' num2str(j) ')'])
   pts = ginput(2);
   pts = pts(:,1) / frameRate;
   wavPts = round(pts * Fs);
   
   soundsc(wav(wavPts(1):wavPts(2),:), Fs)
   
   a{end+1} = wavPts;
   label{end+1} = 'background';
   
end
    
[fPath, fName, fExt] = fileparts(wavFile);
fid = fopen(csvFile,'a');
for j = 1:length(a)
   aa = strtrim(cellstr(num2str(a{j})));
   str = [fName fExt ',' strjoin(aa(:)',',') ',' label{j}];
   fprintf(fid,'%s\n',str); 
end
fclose(fid);
