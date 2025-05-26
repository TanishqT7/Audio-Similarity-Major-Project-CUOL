import argparse
import os
# import numpy as np
from features.audio_utils import (
    load_audio,
    parse_anomaly_timecodes,
    extract_segments,
    input_timecodes_from_user
)
from features.extract_features import(
    compute_mfcc,
    sliding_window_feature,
    aggregate_segment_mfcc
)
from visualization.plot_waveform import plot_waveform
from pydub import AudioSegment
from detection.matcher import match_anomalies
from detection.silence_editor import silence_range_pydub


def parse_args():
    parser = argparse.ArgumentParser(description="Audio Anomaly Remover")

    parser.add_argument("--input", required=True, help="Path to the input audio file")
    parser.add_argument("--timecodes", help="Path to the files containing the timecodes")
    parser.add_argument("--output_dir", default="data/outputs", help="Output Directory Path")

    return parser.parse_args()

def main():
    args = parse_args()

    signal, sr = load_audio(args.input)
    print(f"[INFO] Loaded audio with sample rate {sr}, Length: {len(signal)} samples." )
    plot_waveform(signal, sr, title = "Input Audio - Visual Inspection!")


    if args.timecodes:
        timecodes = parse_anomaly_timecodes(args.timecodes)
        print(f"[INFO] Loaded {len(timecodes)} timecode entries from file.")

    else:
        timecodes = input_timecodes_from_user()
        print(f"[INFO] Collected {len(timecodes)} timecode entries from user input.")

    anomaly_clips = extract_segments(signal, sr, timecodes)
    print(f"[INFO] Extracted {len(anomaly_clips)} anomaly segments.")

    anomaly_feats = []
    for i, clips in enumerate(anomaly_clips, start=1):
        mfcc = compute_mfcc(clips, sr)
        agg = aggregate_segment_mfcc(mfcc)
        anomaly_feats.append(agg)
        print(f"[INFO] Anomaly Clip {i}: MFCC Matrix {mfcc.shape}, aggregated vector {agg.shape}.")



    full_feats, centers = sliding_window_feature(signal, sr, window_sec=0.25, hop_sec=0.125)

    print(f"[INFO] Full audio sliced into {len(full_feats)} windows.")
    print(f"[INFO] First window MFCC shape: {full_feats[0].shape}, center at {centers[0]:.2f}s")


    print("[INFO] Matching anomalies with full Audio Window...")

    detected_ranges = match_anomalies(anomaly_feats, full_feats, centers, window_sec=0.5, threshold=0.99)
    print(f"[INFO] Detected {len(detected_ranges)} anamoly ranges:")

    all_ranges = sorted(timecodes + detected_ranges, key = lambda x: x[0])
    merged = []

    for start, end in all_ranges:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    print(f"[INFO] Ranges to silence: {merged}")

    orig_audio = AudioSegment.from_file(args.input)
    cleaned = silence_range_pydub(orig_audio, merged)

    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]
    out_path = os.path.join(args.output_dir, f"{base}_modified.wav")
    cleaned.export(out_path, format="wav")

    print(f"[INFO] Audio saved to {out_path}")

if __name__ == "__main__":
    main()