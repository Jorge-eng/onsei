# ./wavrunner_all_epochs.sh inDir modelTag outTag

inDir=$1
modelTag=$2
outTag=$3
tmpDir=~/keyword/$3
kws=$4

mkdir -p tmpDir
rm $tmpDir/*.mat

for fn in `ls ../net/models/$modelTag\_ep*.c | sort -u`; do
    echo $fn
    modelName=`echo $fn | xargs -I{} basename {} | cut -d. -f1`
    ./batchrunner.sh $inDir $tmpDir $fn
    python ../dataprep/collect_wavrunner_files.py $tmpDir prob_$modelName
done

python ../dataprep/collect_wavrunner_epochs.py $tmpDir prob_$outTag $kws

