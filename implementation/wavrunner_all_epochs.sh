# ./wavrunner_all_epochs.sh audioFile modelTag outDir

for fn in `ls $2_ep*.c | sort -u`; do
    echo $fn
    ln -sf $fn model_def.c
    wavName=`echo $1 | xargs -I{} basename {} | cut -d. -f1`
    modelName=`echo $fn | xargs -I{} basename {} | cut -d. -f1`
    outName=$wavName\_$modelName.csv
    make clean wavrunner 1> /dev/null
    ./wavrunner $1 > $3/$outName
done

