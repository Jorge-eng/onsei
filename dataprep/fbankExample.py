
import pickle
import audioproc

logM = audioproc.wav2fbank('kwClip_1.wav')
print('logM shape:',logM.shape)

f = open('logM_kwClip_1.pkl', 'wb')
pickle.dump(logM, f)

