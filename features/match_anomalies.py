import numpy as np
from scipy.spatial.distance import cosine
from features.audio_feature import extract_features

def sliding_window(signal, sr, window_sec=0.1, hop_sec=0.05):
    window_size = int(window_sec * sr)
    hop_size = int(hop_sec * sr)

    segments = []

    for start in range(0, len(signal) - window_size, hop_size):
        end = start + window_size
        segments.append((start, end, signal[start:end]))

    return segments

def avg_feat(feat):
    return np.mean(feat, axis=0)

def find_similar_segments(signal, sr, anamoly_feat, threshold = 0.05):

    print("Searching for similar segments in the audio...")

    segments = sliding_window(signal, sr)
    similar_range = []

    anamoly_vector = [avg_feat(a) for a in anamoly_feat]

    for start, end, window in segments:
        try:
            window_feat = extract_features(window, sr)
            window_vector = avg_feat(window_feat)

            for anamoly_vec in anamoly_vector:
                sim = 1 - cosine(anamoly_vec, window_vector)

                if sim > (1 - threshold):
                    similar_range.append((start, end))
                    break

        except Exception as e:
            print(f"Error processing segment {start}-{end}: {e}")

    print(f"Found {len(similar_range)} similar segments.")
    return similar_range