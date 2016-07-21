# Usage:
# ./train.sh inTag modelDef modelType

inTag=_$1
inFile=trainingFeats$inTag\.mat

echo "python train_spec.py $inFile $2 $3 $inTag"
python train_spec.py $inFile $2 $3 $inTag

cd models
./zip_model.sh $2$inTag
cd -

