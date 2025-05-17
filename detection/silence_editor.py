import numpy as np

def silence_ranges(signal: np.ndarray, sr: int, ranges: list) -> np.ndarray:

    sig = signal.copy()

    for start_sec, end_sec in ranges:
        start_idx = int(start_sec * sr)
        end_idx = int(end_sec * sr)

        sig[start_idx:end_idx] = 0

    return sig