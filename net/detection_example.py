from scipy.io import wavfile, savemat
import pdb
import data

cnnModelDef = 'models/cnn_spec_try.json'
cnnModelWeights = 'models/cnn_spec_try.h5'
model = data.load_model(cnnModelDef, cnnModelWeights)

featureFile = 'featureStream1.mat'
features = data.load_batch(featureFile, 'features')
features = (features + 6) / 10
prob1 = model.predict_proba(features, batch_size=128, verbose=1)

featureFile = 'featureStream2.mat'
features = data.load_batch(featureFile, 'features')
features = (features + 6) / 10
prob2 = model.predict_proba(features, batch_size=128, verbose=1)

featureFile = 'featureStream3.mat'
features = data.load_batch(featureFile, 'features')
features = (features + 6) / 10
prob3 = model.predict_proba(features, batch_size=128, verbose=1)

featureFile = 'featureStream4.mat'
features = data.load_batch(featureFile, 'features')
features = (features + 6) / 10
prob4 = model.predict_proba(features, batch_size=128, verbose=1)

featureFile = 'featureStream5.mat'
features = data.load_batch(featureFile, 'features')
features = (features + 6) / 10
prob5 = model.predict_proba(features, batch_size=128, verbose=1)

savemat('prob.mat', {'prob1':prob1, 'prob2':prob2, 'prob3':prob3, 'prob4':prob4, 'prob5':prob5})

