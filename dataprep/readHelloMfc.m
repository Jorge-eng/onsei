function readHelloMfc(indir, outFile)

addpath('/Users/hello/david/matlab')

%indir = '/Users/hello/david/data/esc_augmented_wav/wavs/pos';
fileType = 'mat';

allFiles = findFiles(indir,fileType);

mfc = {};
c = 0;
for j = 1:length(allFiles)
    
    data = load(fullfile(indir, allFiles{j}));
    if size(data.arr2, 1) >= 1664
        c = c+1;
        mfc{c} = data.arr2(1:1664,:)';
    else
        size(data.arr2, 1)
    end
end

mfc = cat(3, mfc{:});

save(outFile, 'mfc')