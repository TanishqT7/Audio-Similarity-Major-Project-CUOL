import librosa
import numpy as np

def extract_mfcc(signal, sr, n_mfcc=13, n_fft=2048, hop_length=512):

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    return mfcc.T

def samples_to_time(start, end, sr):
    return start / sr, end / sr