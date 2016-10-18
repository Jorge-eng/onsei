
modelName=$1
mkdir -p ../net/outputs/$modelName

#testDirs="reverberant_speech/16k"

#testDirs="testingWavs_kwClip_okay_sense
#testingWavs_kwClip_stop
#testingWavs_kwClip_snooze
#noiseDataset/16k
#reverberant_speech/16k"

testDirs="testingWavs_kwClip_okay_sense
testingWavs_kwClip_stop
testingWavs_kwClip_snooze
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
    mkdir -p ../net/outputs/$modelName/$td
    ../implementation/batchrunner.sh ~/keyword/$td ../net/outputs/$modelName/$td ../net/models/$modelName\.c 
done
