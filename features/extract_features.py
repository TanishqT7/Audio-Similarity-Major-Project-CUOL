import numpy as np
import librosa

def compute_mfcc(signal: np.ndarray,
        sr: int,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512) -> np.ndarray:
    
    if signal.dtype != np.float32:
        signal = signal.astype(np.float32)
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))
    
    mfcc = librosa.feature.mfcc(y=signal,
                                sr=sr,
                                n_mfcc=n_mfcc,
                                hop_length=hop_length,
                                n_fft=n_fft)
    
    # mfcc = librosa.util.normalize(mfcc, axis=1)
    return mfcc


def sliding_window_feature(signal: np.ndarray,
                           sr: int,
                           window_sec: float = 1.0,
                           hop_sec: float = 0.5,
                           feature_fn = compute_mfcc,
                           **feat_kwargs) -> tuple[list[np.ndarray], np.ndarray]:
    
    window_len = int(window_sec * sr)
    hop_len = int(hop_sec * sr)

    features = []
    centers = []

    for start in range(0, len(signal) - window_len+1, hop_len):
        end = start + window_len
        seg = signal[start:end]

        feat = feature_fn(seg, sr, **feat_kwargs)
        features.append(feat)

        centers.append((start + window_len / 2) / sr)

    return features, np.array(centers)


def aggregate_segment_mfcc(mfcc: np.ndarray) -> np.ndarray:
    return np.mean(mfcc, axis=1)