import numpy as np
from scipy.spatial.distance import cosine

def match_anomalies(anomaly_vecs, full_feats, centers, window_sec=1.0, threshold=0.75):

    full_aggs = [np.mean(feat, axis=1) for feat in full_feats]

    matches = []

    for a_vec in anomaly_vecs:
        for idx, f_vec in enumerate(full_aggs):
            sim = 1 - cosine(a_vec, f_vec)
            if sim >= threshold:
                center = centers[idx]
                start = center - window_sec / 2
                end = center + window_sec / 2
                matches.append((max(0, start), end))

    if not matches:
        return []
    
    matches.sort(key=lambda x: x[0])
    merged = [matches[0]]

    for curr in matches[1:]:
        prev = merged[-1]

        if curr[0] <= prev[1]:
            merged[-1] = (prev[0], max(prev[1], curr[1]))

        else:
            merged.append(curr)

    return merged