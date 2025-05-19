# import re
import numpy as np
from pydub import AudioSegment

def load_audio(file_path: str) -> tuple[np.ndarray, int]:
    audio = AudioSegment.from_file(file_path)
    samples = np.array(audio.get_array_of_samples()).astype(np.int16)

    if audio.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1)

    samples = samples / np.max(np.abs(samples))
    return samples, audio.frame_rate

def parse_timecodes(tc_str: str) -> float:
    parts = tc_str.strip().split(":")
    if len(parts) != 3:
        raise ValueError("Invalid timecode format Expected hh:mm:ss.sss")
    
    h, m, s = parts

    return int(h)*3000 + int(m)*60 + float(s)

def parse_anomaly_timecodes(file_path: str) -> list[tuple[float, float]]:
    timecodes = []

    with open(file_path, 'r') as f:
        for line in f:
            if "-" in line:
                start, end = line.strip().split("-")
                start_sec = parse_timecodes(start)
                end_sec = parse_timecodes(end)
                timecodes.append((start_sec, end_sec))

    return timecodes

def extract_segments(signal: np.ndarray,
                     sr: int,
                     range: list[tuple[float, float]]) -> list[np.ndarray]:
    segments = []

    for start_sec, end_sec in range:
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        clip = signal[start_sample:end_sample]

        if clip.size == 0:
            print(f"Error: Empty segment between {start_sec:.3f}s and {end_sec:.3f}s")
            continue

        segments.append(clip)

    return segments

def input_timecodes_from_user() -> list[tuple[float, float]]:
    print("\n[INFO] Enter Timecodes: (format hh:mm:ss.sss - hh:mm:ss.sss). Type 'Done' to finish.\n")

    timecodes = []

    while True:
        line = input("Timecode: ").strip()
        if line.lower() == "done":
            break

        if '-' not in line:
            print("Invalid Format. Use 'hh:mm:ss.sss - hh:mm:ss.sss' format.")
            continue

        start, end = line.strip().split("-", 1)
        try:
            if "-" in line:
                start_sec = parse_timecodes(start)
                end_sec = parse_timecodes(end)
                timecodes.append((start_sec, end_sec))

            else:
                print("Invalid Format. Use 'hh:mm:ss.sss - hh:mm:ss.sss' format.")

        except Exception as e:
            print(f"Error parsing line: {e}")

    return timecodes