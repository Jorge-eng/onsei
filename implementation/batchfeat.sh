find $1 -type f -iname \*.wav | xargs -I {} basename {} | xargs -I {} ./tinyfeats $1/{} $2/{}.bin
