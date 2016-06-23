
# Python features
## Clean
### Training
./wav2feats.sh train ~/Dropbox/Data/keyword/trainingWavs trainingFeats
### Testing  
./wav2feats.sh test ~/Dropbox/Data/keyword/testingWavs testingFeats.mat
## Noisy
### Training
./wav2feats.sh train ~/Dropbox/Data/keyword/trainingWavsNoise trainingFeatsNoise
### Testing
./wav2feats.sh test ~/Dropbox/Data/keyword/testingWavsNoise testingFeatsNoise.mat

# tinyfeats
## Clean
### Training
./wav2tinyfeats.sh train ~/Dropbox/Data/keyword/trainingWavs ~/Dropbox/Data/keyword/trainingFeats trainingTinyfeats
### Testing  
./wav2tinyfeats.sh test ~/Dropbox/Data/keyword/testingWavs ~/Dropbox/Data/keyword/testingFeats testingTinyfeats.mat
## Noisy
### Training
./wav2tinyfeats.sh train ~/Dropbox/Data/keyword/trainingWavsNoise ~/Dropbox/Data/keyword/trainingFeatsNoise trainingTinyfeatsNoise
### Testing  
./wav2tinyfeats.sh test ~/Dropbox/Data/keyword/testingWavsNoise ~/Dropbox/Data/keyword/testingFeatsNoise testingTinyfeatsNoise.mat

