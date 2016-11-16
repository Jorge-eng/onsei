# Example: 
# $ ./wav2feats.sh [train | test] [py | tiny] kw1+kw2+.+kwN tag

dataRoot=~/keyword
inDir=$dataRoot/$3/$1/wav
outName=$1\ingFeats_$3\_$2\_$4

if [ $2 = "tiny" ]; then
    outFeaDir=$dataRoot/$3/$1/bin
    if [ ! -d "$outFeaDir" ]; then
        echo "batchfeat.sh $inDir $outFeaDir"
        ../implementation/batchfeat.sh $inDir $outFeaDir
    fi
    inDir=$outFeaDir
fi

if [ $1 = "train" ]; then
    echo "createTrainingFeatures.py $inDir $outName $3"
    python createTrainingFeatures.py $inDir $outName $3

    echo "copyToHelloServer.sh $outName\.mat onsei/net"
    ~/copyToHelloServer.sh $outName\.mat onsei/net
elif [ $1 = "test" ]; then
    echo "createTestGrid.py $inDir $outName $3" 
    python createTestGrid.py $inDir $outName $3 #${@:4}
fi

