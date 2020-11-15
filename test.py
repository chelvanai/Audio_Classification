import tensorflow as tf
import librosa
import numpy as np
import math

# model = tf.keras.models.load_model('music_genres_model.h5')

signal, sample_rate = librosa.load('classical_test.wav', sr=22050)

print(signal.shape)

mfcc = librosa.feature.mfcc(signal, sample_rate, n_mfcc=13, n_fft=2048,
                            hop_length=512)

print(mfcc.shape)