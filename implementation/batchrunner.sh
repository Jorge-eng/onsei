# ./batchrunner inDir outDir

inDir=$1
outDir=$2

mkdir -p $outDir

find $inDir -type f -iname \*.wav | sort | xargs -I{} basename {} | xargs -I{} -P4 sh -c "../implementation/wavrunner '$inDir/{}' > '$outDir/{}.csv'"

