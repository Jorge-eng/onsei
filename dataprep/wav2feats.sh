# Example: 
# $ ./wav2feats.sh [train | test] [py | tiny] kw1+kw2+...

dataRoot=~/Dropbox/Data/keyword
inDir=$dataRoot/$1\ingWavs
outName=$1\ingFeats_$3\_$2

if [ $2 = "tiny" ]; then
    outFeaDir=$dataRoot/$1\ingFeats
    if [ ! -d "$outFeaDir" ]; then
        echo "batchfeat.sh $inDir $outFeaDir"
        ../implementation/batchfeat.sh $inDir $outFeaDir
    fi
    inDir=$outFeaDir
fi

if [ $1 = "train" ]; then
    echo "createTrainingFeatures.py $inDir $outName $3"
    python createTrainingFeatures.py $inDir $outName $3

    echo "copyToHelloServer.sh $outName.mat onsei/net"
    ~/copyToHelloServer.sh $outName\.mat onsei/net
elif [ $1 = "test" ]; then
    echo "createTestGrid.py $inDir $outName $3" 
    python createTestGrid.py $inDir $outName $3 #${@:4}
fi

