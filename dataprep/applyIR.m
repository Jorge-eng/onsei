function wav = applyIR(wav, IR)

L = length(wav);
wav = conv(IR, wav);
wav = wav(1:L);
