# ./batchrunner inDir inMatcher outDir

inDir=$1
matcher=$2
outDir=$3

mkdir -p $outDir

find $inDir -type f -iname $matcher*.wav | xargs -I{} basename {} | xargs -I{} -P4 sh -c "../implementation/wavrunner '$inDir/{}' > '$outDir/{}.csv'"

