# Example: 
# $ ./wav2feats.sh train ~/Dropbox/Data/keyword/trainingWavs trainingFeats kw1 kw2 ...
# $ ./wav2feats.sh test ~/Dropbox/Data/keyword/testingWavs testingFeats kw1 kw2 ...

if [ $1 = "train" ]; then
    echo "createTrainingFeatures.py"
    python createTrainingFeatures.py $2 $3 ${@:4}

    echo "copyToHelloServer.sh"
    ~/copyToHelloServer.sh $3\.mat onsei/net
elif [ $1 = "test" ]; then
    echo "createTestGrid.py"
    python createTestGrid.py $2 $3 ${@:4}
fi

