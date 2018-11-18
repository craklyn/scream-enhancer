import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

import pickle

import build_landmarks
from collections import Counter
import time

fs=48000
duration = 20  # seconds
timeWindow = duration

frames_per_timeWindow = duration * fs / (build_landmarks.DEFAULT_WINDOW_SIZE * build_landmarks.DEFAULT_OVERLAP_RATIO)

with open("scream_landmarks_table.pickle", "rb") as counter_pickle_file:
    landmarks_table = pickle.load(counter_pickle_file)

while True:
    myrecording = sd.rec(duration * fs, samplerate=fs, channels=1, dtype='int16')
    print("Recording Audio")
    sd.wait()
    #print("Audio recording complete , Play Audio")
    #sd.play(myrecording, fs)
    #sd.wait()
    #print("Play Audio Complete")

    feature_generator = build_landmarks.fingerprint([x[0] for x in myrecording], Fs=48000)

    t1_diffs = Counter()
    for rec_f, rec_t1 in feature_generator:
        rec_t1 = int(rec_t1 * timeWindow / frames_per_timeWindow)  # Express t1 in seconds since start of file

        if rec_f in landmarks_table:
            t1s = landmarks_table[rec_f]
            for t1 in t1s:
                t1_diffs[int(t1 - rec_t1)] += 1

    print(t1_diffs.most_common(1)[0])
    time_s = t1_diffs.most_common(1)[0][0]
    print("Time: " + time.strftime('%H:%M:%S', time.gmtime(time_s)))

    #break



