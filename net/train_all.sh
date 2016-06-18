
model=model_jun17_small_sigm

unzip trainingFeats.zip 
python train_spec.py $model cnn
cd models
./zip_model.sh model_jun17_small_sigm _clean
cd -

unzip trainingFeatsNoise.zip 
python train_spec.py $model cnn
cd models
./zip_model.sh model_jun17_small_sigm _noise
cd -

unzip trainingTinyfeats.zip 
python train_spec.py $model cnn
cd models
./zip_model.sh model_jun17_small_sigm _clean_tiny
cd -

unzip trainingTinyfeatsNoise.zip 
python train_spec.py $model cnn
cd models
./zip_model.sh model_jun17_small_sigm _noise_tiny
cd -

