# Usage:
# ./predict.sh inTag modelDef

echo "python predict_spec.py features ../dataprep/testingFeats_$1.mat prob_$2_$1 $2_$1"
python predict_spec.py features ../dataprep/testingFeats_$1\.mat prob_$2\_$1 $2\_$1

