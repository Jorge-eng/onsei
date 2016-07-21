# Usage:
# ./train.sh inTag modelDef modelType normalize

inTag=_$1
inFile=trainingFeats$inTag\.mat

echo "python train_spec.py $inFile $2 $3 $inTag $4"
python train_spec.py $inFile $2 $3 $inTag $4

cd models
./zip_model.sh $2$inTag
cd -

