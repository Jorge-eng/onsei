mkdir -p $2
cp $1/*.csv $2

find $1 -maxdepth 1 -type f -iname \*.wav | xargs -I{} basename {} | xargs -I{} -P4 ../implementation/tinyfeats $1/{} $2/{}.bin
