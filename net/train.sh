# Usage:
# ./train.sh inTag outTag modelDef modelType normalize w1,w2,...

inTag=_$1
outTag=_$2
inFile=trainingFeats$inTag\.mat

echo "python train_spec.py $inFile $3 $4 $inTag$outTag $5 $6"
python train_spec.py $inFile $3 $4 $inTag$outTag $5 $6

cd models
./zip_model.sh $3$inTag$outTag
cd -

