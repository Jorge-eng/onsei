# ./wavrunner_all_epochs.sh inDir matcher modelTag outTag

inDir=$1
matcher=$2
modelTag=$3
outTag=$4
tmpDir=~/keyword/$4

mkdir -p tmpDir
rm $tmpDir/*.csv $tmpDir/*.mat

for fn in `ls ../net/models/$modelTag_ep*.c | sort -u`; do
    echo $fn
    modelName=`echo $fn | xargs -I{} basename {} | cut -d. -f1`
    ln -sf $fn model_def.c
    make clean wavrunner 1> /dev/null
    ./batchrunner.sh $inDir $matcher $tmpDir
    python ../dataprep/collect_wavrunner_files.py $tmpDir prob_$modelName
done

python ../dataprep/collect_wavrunner_epochs.py $tmpDir prob_$outTag

