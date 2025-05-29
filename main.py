import argparse
import os
import csv
import numpy as np
from collections import OrderedDict
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
from pydub.utils import make_chunks
from detection.matcher import match_anomalies
from detection.silence_editor import silence_range_pydub


def parse_args():
    parser = argparse.ArgumentParser(description="Audio Anomaly Remover")

    parser.add_argument("--input", required=True, help="Path to the input audio file")
    parser.add_argument("--timecodes", help="Path to the files containing the timecodes")
    parser.add_argument("--output_dir", default="data/outputs", help="Output Directory Path")
    parser.add_argument("--chunk_minutes", type=int, default=5, help="Chunk size in minutes")

    parser.add_argument("--label", action="store_true", help="Append Window Labels to CSV (does no exist)")
    parser.add_argument("--make_dataset", action="store_true", help="Build training_data.npz from windows_labels.csv")
    parser.add_argument("--window_sec", default= 0.5, type=float, help="Window length (seconds) for Labeling")
    parser.add_argument("--hop_sec", default= 0.25, type=float, help="Hop length (seconds) for Labeling")

    return parser.parse_args()

def process_chunk(chunk, chunk_idx, chunk_start_sec, anamoly_feats, manual_ranges, sr):

    samples = np.array(chunk.get_array_of_samples()).astype(np.int16)
    if chunk.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1).astype(np.int16)

    full_feats, centers = sliding_window_feature(samples, sr, window_sec=0.25, hop_sec=0.125)
    detected_ranges = match_anomalies(anamoly_feats, full_feats, centers, window_sec=0.5, threshold=0.99)

    chunk_end = chunk_start_sec + len(chunk) / 1000
    manual_in_chunk = [
        (max(0, s - chunk_start_sec), min(e - chunk_start_sec, len(chunk) / 1000))
        for s, e in manual_ranges
        if s < chunk_end and e > chunk_start_sec
    ]

    all_ranges = sorted(detected_ranges + manual_in_chunk, key=lambda x: x[0])
    merged = []

    for start, end in all_ranges:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    print(f"[INFO] Silencing {len(merged)} merged ranges in chunk {chunk_idx + 1}.")
    cleaned_chunk = silence_range_pydub(chunk, merged)

    return cleaned_chunk, len(merged)

def label_windows(audio_path, timecodes, window_sec, hop_sec, chunk_minutes=5):

    audio = AudioSegment.from_file(audio_path)
    chunk_ms_len = chunk_minutes * 60 * 1000
    chunks = make_chunks(audio, chunk_ms_len)

    base_name = os.path.basename(audio_path)
    out_csv = "data/window_labels.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    label_dict = OrderedDict()
    if os.path.exists(out_csv):
        with open(out_csv, newline="") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                key = (row["file_name"], int(row["window_idx"]))
                label_dict[key] = row

    sr = audio.frame_rate
    global_idx = 0

    for chunk_idx, chunk in enumerate(chunks):
        start_sec = (chunk_idx * chunk_ms_len) / 1000
        samples = np.array(chunk.get_array_of_samples()).astype(np.int16)
        if chunk.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)

        feats, centers = sliding_window_feature(samples, sr, window_sec=window_sec, hop_sec=hop_sec)

        for c in centers:

            global_center = start_sec + c
            w_start, w_end = global_center - window_sec/2, global_center + window_sec/2
            label = int(any(s < w_end and e > w_start for s, e in timecodes))
            key = (base_name, global_idx)
            label_dict[key] = {
                "file_name": base_name,
                "file_path": audio_path,
                "window_idx": global_idx,
                "start_sec": f"{w_start:.3f}",
                "end_sec": f"{w_end:.3f}",
                "label": label
            }
            global_idx += 1

    with open(out_csv, "w", newline="") as f:
        fieldnames = ["file_name", "file_path", "window_idx", "start_sec", "end_sec", "label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in label_dict.values():
            writer.writerow(row)


    print(f"[INFO] window_labels.csv updated with {len(label_dict)} unique rows.")


def build_dataset(audio_file, window_sec, hop_sec):

    audio = AudioSegment.from_file(audio_file)

    X, Y = [], []
    with open("data/window_labels.csv") as f:
        rdr = csv.DictReader(f)

        audio = AudioSegment.from_file(audio_file)
        for row in rdr:

            start_ms = int(float(row["start_sec"]) * 1000)
            end_ms = int(float(row["end_sec"]) * 1000)
            label = int(row["label"])

            segment = audio[start_ms:end_ms]

            samples = np.array(segment.get_array_of_samples()).astype(np.float32)
            if audio.channels == 2:
                samples = samples.reshape((-1, 2)).mean(axis=1)
            
            sr = segment.frame_rate

            mfcc = compute_mfcc(samples, sr)
            vec = aggregate_segment_mfcc(mfcc)

            X.append(vec)
            Y.append(label)

    X = np.stack(X)
    Y = np.array(Y)

    np.savez_compressed("data/training_data.npz", X=X, Y=Y)
    print(f"[INFO] Dataset saved to data/training_data.npz: X.shape = {X.shape}, Y.shape = {Y.shape}")
        
def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]
    
    full_signal, sr = load_audio(args.input)
    print(f"[INFO] Loaded audio with sample rate {sr}, Length: {len(full_signal)} samples." )
    #plot_waveform(full_signal, sr, title = "Input Audio - Visual Inspection!")

    orig_audio = AudioSegment.from_file(args.input)
    chunk_ms_len = args.chunk_minutes * 60 * 1000
    chunks = make_chunks(orig_audio, chunk_ms_len)

    if args.timecodes:
        timecodes = parse_anomaly_timecodes(args.timecodes)
        print(f"[INFO] Loaded {len(timecodes)} timecode entries from file.")

    else:
        timecodes = input_timecodes_from_user()
        print(f"[INFO] Collected {len(timecodes)} timecode entries from user input.")

    if args.label:
        label_windows(args.input, timecodes, args.window_sec, args.hop_sec)

    if args.make_dataset:
        build_dataset(args.input, args.window_sec, args.hop_sec)
        
    anomaly_clips = extract_segments(full_signal, sr, timecodes)
    anomaly_feats = [aggregate_segment_mfcc(compute_mfcc(clip, sr)) for clip in anomaly_clips]
    print(f"[INFO] Extracted and computed features for {len(anomaly_feats)} anomaly segments.")

    cleaned_chunks = []
    total_silenced_clips = 0
    for idx, chunk in enumerate(chunks):
        chunk_start_sec = (idx * chunk_ms_len) / 1000.0
        cleaned, num_ranges = process_chunk(chunk, idx, chunk_start_sec, anomaly_feats, timecodes, sr)
        cleaned_chunks.append(cleaned)
        total_silenced_clips += num_ranges

    final_audio = sum(cleaned_chunks[1:], cleaned_chunks[0])
    out_path = os.path.join(args.output_dir, f"{base}_modified.wav")
    final_audio.export(out_path, format="wav")

    print(f"[INFO] Total ranges silenced: {total_silenced_clips}")
    print(f"[INFO] Audio saved to {out_path}")

if __name__ == "__main__":
    main()