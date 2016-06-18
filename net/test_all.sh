
model=model_jun17_small_sigm
ngrokport=17101

scp -P $ngrokport ubuntu@0.tcp.ngrok.io:onsei/net/models/$model\_clean.zip models/
scp -P $ngrokport ubuntu@0.tcp.ngrok.io:onsei/net/models/$model\_noise.zip models/
scp -P $ngrokport ubuntu@0.tcp.ngrok.io:onsei/net/models/$model\_clean_tiny.zip models/
scp -P $ngrokport ubuntu@0.tcp.ngrok.io:onsei/net/models/$model\_noise_tiny.zip models/

cd models ; unzip -o $model\_clean.zip ; cd -
python predict_spec.py features ../dataprep/testingFeats.mat prob_cleanData_cleanModel.mat $model
python predict_spec.py features ../dataprep/testingFeatsNoise.mat prob_noiseData_cleanModel.mat $model

cd models ; unzip -o $model\_noise.zip ; cd -
python predict_spec.py features ../dataprep/testingFeats.mat prob_cleanData_noiseModel.mat $model
python predict_spec.py features ../dataprep/testingFeatsNoise.mat prob_noiseData_noiseModel.mat $model

cd models ; unzip -o $model\_clean_tiny.zip ; cd -
python predict_spec.py features ../dataprep/testingTinyfeats.mat prob_cleanData_cleanModel_tiny.mat $model
python predict_spec.py features ../dataprep/testingFeatsNoise.mat prob_noiseData_cleanModel_tiny.mat $model

cd models ; unzip -o $model\_noise_tiny.zip ; cd -
python predict_spec.py features ../dataprep/testingTinyfeats.mat prob_cleanData_noiseModel_tiny.mat $model
python predict_spec.py features ../dataprep/testingTinyfeatsNoise.mat prob_noiseData_noiseModel_tiny.mat $model

