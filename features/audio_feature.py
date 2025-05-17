import librosa
import numpy as np

def extract_features(signal, sr, n_fft=2048, hop_length=512, n_mels=128):

    feat = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    return feat.T

def samples_to_time(start, end, sr):
    return start / sr, end / sr