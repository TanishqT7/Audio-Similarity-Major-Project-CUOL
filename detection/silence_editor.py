# import numpy as np
from pydub import AudioSegment

'''
def silence_ranges(signal: np.ndarray, sr: int, ranges: list) -> np.ndarray:

    sig = signal.copy()

    for start_sec, end_sec in ranges:
        start_idx = int(start_sec * sr)
        end_idx = int(end_sec * sr)

        sig[start_idx:end_idx] = 0

    return sig
'''

def silence_range_pydub(audio: AudioSegment,
                        ranges: list[tuple[float, float]]) -> AudioSegment:
    
    out = AudioSegment.empty()
    prev_end_ms = 0
    total_ms = len(audio)

    for start_sec, end_sec in sorted(ranges, key = lambda x: x[0]):
        start_ms = max(0, int(start_sec * 1000))
        end_ms = min(total_ms, int(end_sec * 1000))

        out += audio[prev_end_ms:start_ms]

        out+= AudioSegment.silent(duration=end_ms - start_ms)

        prev_end_ms = end_ms

    out += audio[prev_end_ms:]

    return out