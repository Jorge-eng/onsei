# Usage ./testModelPy.sh model_name epoch [make_input]

modelName=$1
mkdir -p ../net/outputs/$modelName\_ep$2

#testDirs="reverberant_speech/16k"
#testDirs="TED"

#testDirs="testingWavs_kwClip_okay_sense
#testingWavs_kwClip_stop
#testingWavs_kwClip_snooze
#noiseDataset/16k
#reverberant_speech/16k"

testDirs="testingWavs_kwClip_okay_sense
noiseDataset/16k
reverberant_speech/16k
TED"

#testingWavs_adversarial

#testDirs="testingWavs_kwClip_okay_sense
#testingWavs_kwClip_stop
#testingWavs_kwClip_snooze
#noiseDataset/16k"

##speechDataset/16k"

for td in $testDirs; do
    echo $td
    mkdir -p ../net/outputs/$modelName\_ep$2/$td
    if [ $3 = "make_input" ]; then
        mkdir -p ~/keyword/$td/bin
        ../implementation/batchfeat.sh ~/keyword/$td ~/keyword/$td/bin
    fi
    python ../net/predict_all_files.py ~/keyword/$td/bin ../net/outputs/$modelName\_ep$2/$td $modelName $2
done

