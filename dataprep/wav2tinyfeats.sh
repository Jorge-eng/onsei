# Example: 
# $ ./wav2tinyfeats.sh train ~/Dropbox/Data/keyword/trainingWavs ~/Dropbox/Data/keyword/trainingFeats trainingFeats
# $ ./wav2tinyfeats.sh test ~/Dropbox/Data/keyword/testingWavs ~/Dropbox/Data/keyword/testingFeats testingFeats

echo "batchfeat.sh"
../implementation/batchfeat.sh $2 $3

./wav2feats.sh $1 $3 $4 ${@:5}

