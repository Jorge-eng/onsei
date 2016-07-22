# Usage:
# ./predict.sh inTag modelDef [epoch # | auto] [get]

if [ ! -z "$3" ]; then
    epoch=$3
    if [ $epoch = "auto" ]; then
        epoch=
    fi
fi

if [ ! -z "$4" ] && [ $4 = "get" ]; then
    ./get_model.sh $2\_$1
fi

echo "python predict_spec.py features ../dataprep/testingFeats_$1.mat prob_$2_$1 $2_$1 $epoch"
python predict_spec.py features ../dataprep/testingFeats_$1\.mat prob_$2\_$1 $2\_$1 $epoch

