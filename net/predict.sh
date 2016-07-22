# Usage:
# ./predict.sh trTag modelDef modelTag [epoch # | auto] [get]

if [ ! -z "$4" ]; then
    epoch=$4
    if [ $epoch = "auto" ]; then
        epoch=
    fi
fi

modelTag=$3
if [ ! -z "$modelTag" ]; then
    modelTag=_$modelTag 
fi

model=$2_$1$modelTag
if [ ! -z "$5" ] && [ $5 = "get" ]; then
    ./get_model.sh $model
fi

echo "python predict_spec.py features ../dataprep/testingFeats_$1.mat prob_$model $model $epoch"
python predict_spec.py features ../dataprep/testingFeats_$1.mat prob_$model $model $epoch

