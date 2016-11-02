# ./batchrunner inDir outDir modelTag

inDir=$1
outDir=$2
modelTag=$3
modelTag=`echo $modelTag | sed "s/+/_/g"`

mkdir -p $outDir
rm $outDir/*.csv

cd ../implementation

ln -sf ../net/models/$modelTag\.c model_def.c
touch wavrunner.cpp
make wavrunner 1> /dev/null

find $inDir -type f -iname \*.wav | sort | xargs -I{} basename {} | xargs -I{} -P4 sh -c "./wavrunner '$inDir/{}' > '$outDir/{}.csv'"

cd -
