find ../data/esc_augmented_wav/wavs/pos -type f -name \*.wav | xargs -I{} -P8 ./wavrunner {}
find ../data/esc_augmented_wav/wavs/pos -type f -name \*.proto | xargs -I{} -P8 ./convertFeats.py {}
