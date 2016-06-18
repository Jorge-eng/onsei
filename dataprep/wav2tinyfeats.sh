# Example: 
# $ ./wav2tinyfeats.sh train ~/Dropbox/Data/keyword/trainingWavs ~/Dropbox/Data/keyword/trainingFeats trainingFeats
# $ ./wav2tinyfeats.sh test ~/Dropbox/Data/keyword/testingWavs ~/Dropbox/Data/keyword/testingFeats testingFeats.mat

echo "batchfeat.sh"
../implementation/batchfeat.sh $2 $3

if [ $1 = "train" ]; then
    echo "loadEmbeddedFeatures.py"
    python loadEmbeddedFeatures.py $3

    echo "zip"
    rm -f $4\.zip 
    zip -r $4\.zip spec_pos.mat spec_neg.mat

    echo "copyToHelloServer.sh"
    ~/copyToHelloServer.sh $4\.zip onsei/net
elif [ $1 = "test" ]; then
    echo "createTestGrid.py"
    python createTestGrid.py $3 $4 
fi

