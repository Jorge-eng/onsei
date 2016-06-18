
model=model_jun17_small_sigm

unzip -o trainingFeats.zip ; python train_spec.py $model cnn
cd models ; ./zip_model.sh model_jun17_small_sigm _clean ; cd -

unzip -o trainingFeatsNoise.zip ; python train_spec.py $model cnn
cd models ; ./zip_model.sh model_jun17_small_sigm _noise ; cd -

unzip -o trainingTinyfeats.zip ; python train_spec.py $model cnn
cd models ; ./zip_model.sh model_jun17_small_sigm _clean_tiny ; cd -

unzip -o trainingTinyfeatsNoise.zip ; python train_spec.py $model cnn
cd models ; ./zip_model.sh model_jun17_small_sigm _noise_tiny ; cd -

