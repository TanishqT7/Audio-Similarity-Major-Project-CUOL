import argparse
import os
import numpy as np
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

    return parser.parse_args()

def process_chunk(chunk, chunk_idx, chunk_start_sec, anamoly_feats, sr):

    samples = np.array(chunk.get_array_of_samples()).astype(np.int16)
    if chunk.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1).astype(np.int16)

    full_feats, centers = sliding_window_feature(samples, sr, window_sec=0.25, hop_sec=0.125)
    detected_ranges = match_anomalies(anamoly_feats, full_feats, centers, window_sec=0.5, threshold=0.99)

    all_ranges = sorted(detected_ranges, key=lambda x: x[0])
    merged = []

    for start, end in all_ranges:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    print(f"[INFO] Silencing {len(merged)} merged ranges in chunk {chunk_idx + 1}.")
    cleaned_chunk = silence_range_pydub(chunk, merged)

    return cleaned_chunk, len(merged)


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

    anomaly_clips = extract_segments(full_signal, sr, timecodes)
    anomaly_feats = [aggregate_segment_mfcc(compute_mfcc(clip, sr)) for clip in anomaly_clips]
    print(f"[INFO] Extracted and computed features for {len(anomaly_feats)} anomaly segments.")

    cleaned_chunks = []
    total_silenced_clips = 0
    for idx, chunk in enumerate(chunks):
        chunk_start_sec = (idx * chunk_ms_len) / 1000.0
        cleaned, num_ranges = process_chunk(chunk, idx, chunk_start_sec, anomaly_feats, sr)
        cleaned_chunks.append(cleaned)
        total_silenced_clips += num_ranges

    final_audio = sum(cleaned_chunks[1:], cleaned_chunks[0])
    out_path = os.path.join(args.output_dir, f"{base}_modified.wav")
    final_audio.export(out_path, format="wav")

    print(f"[INFO] Total ranges silenced: {total_silenced_clips}")
    print(f"[INFO] Audio saved to {out_path}")

if __name__ == "__main__":
    main()