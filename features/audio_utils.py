import os
import re
import numpy as np
from pydub import AudioSegment

def load_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    samples = np.array(audio.get_array_of_samples()).astype(np.int16)

    if audio.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1)

    return samples, audio.frame_rate

def parse_timecodes(tc_str):
    parts = re.split(r'[:.]', tc_str.strip())
    if len(parts) < 3:
        raise ValueError("Invalid timecode format Expected hh:mm:ss.sss")
    
    h, m, s = int(parts[0]), int(parts[1]), int(parts[2])

    return h * 3000 + m * 60 + s

def parse_anomaly_timecodes(file_path):
    timecodes = []

    with open(file_path, 'r') as f:
        for line in f:
            if "-" in line:
                start, end = line.strip().split("-")
                start_sec = parse_timecodes(start)
                end_sec = parse_timecodes(end)
                timecodes.append((start_sec, end_sec))

    return timecodes

def extract_segments(signal, sr, range):
    segments = []

    for start_sec, end_sec in range:
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)

        segments.append(signal[start_sample:end_sample])

    return segments

def input_timecodes_from_user():
    print("\n[INFO] Enter Timecodes: (format hh:mm:ss.sss - hh:mm:ss.sss). Type 'Done' to finish.\n")

    timecodes = []

    while True:
        line = input("Timecode: ")
        if line.lower() == "done":
            break

        try:
            if "-" in line:
                start, end = line.strip().split("-")
                start_sec = parse_timecodes(start)
                end_sec = parse_timecodes(end)
                timecodes.append((start_sec, end_sec))

            else:
                print("Invalid Format. Use 'hh:mm:ss.sss - hh:mm:ss.sss' format.")

        except Exception as e:
            print(f"Error parsing line: {e}")

    return timecodes