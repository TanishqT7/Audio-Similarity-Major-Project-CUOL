import numpy as np
from librosa.sequence import dtw
from librosa.util import normalize

def match_anomalies(anomaly_vecs, full_feats, centers, window_sec=0.5, threshold=0.99):

    full_aggs = [np.mean(feat, axis=1) for feat in full_feats]

    matches = []

    for anomaly in anomaly_vecs:
        for idx, window in enumerate(full_aggs):
            dist = dtw_distance(anomaly, window)
            if dist <= threshold:
                center = centers[idx]
                start = center - window_sec / 2
                end = center + window_sec / 2
                matches.append((max(0, start), end))

    if not matches:
        return []
    
    matches.sort(key=lambda x: x[0])
    merged = [matches[0]]
    dtw_distances = []

    for curr in matches[1:]:
        prev = merged[-1]

        if curr[0] <= prev[1]:
            merged[-1] = (prev[0], max(prev[1], curr[1]))

        else:
            merged.append(curr)

        dtw_distances.append(f"{dist:.2f}")
        # print(f"[DEBUG] DTW Distance: {dist:.2f}")
    print("[DEBUG] DTW Distance:", dtw_distances[0])
    return merged

def dtw_distance(query: np.ndarray, target: np.ndarray) -> float:

    query = normalize(query)
    target = normalize(target)

    D, _ = dtw(X=query.T, Y=target.T, metric="euclidean")

    return D[-1, -1]