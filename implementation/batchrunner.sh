# ./batchrunner inDir outDir

inDir=$1
outDir=$2
modelFile=$3

mkdir -p $outDir
rm $outDir/*.csv

cd ../implementation

ln -sf $modelFile model_def.c
touch wavrunner.cpp
make wavrunner 1> /dev/null

find $inDir -type f -iname \*.wav | sort | xargs -I{} basename {} | xargs -I{} -P4 sh -c "./wavrunner '$inDir/{}' > '$outDir/{}.csv'"

cd -
