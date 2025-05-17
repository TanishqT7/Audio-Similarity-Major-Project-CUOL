import argparse
import os
from features.audio_utils import (
    load_audio,
    parse_anomaly_timecodes,
    extract_segments,
    input_timecodes_from_user
)
from visualization.plot_waveform import plot_waveform
from pydub import AudioSegment
from features.audio_feature import (
    extract_features,
    samples_to_time
)
from features.match_anomalies import find_similar_segments

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

    anomaly_features = [extract_features(clip, sr) for clip in anomaly_clips]

    print(f"[INFO] Extracted Mel Spectrogram from {len(anomaly_features)} anomaly clips.")
    for i, f in enumerate(anomaly_features):
        print(f" - Clip {i + 1}: {f.shape} (frames x coeffs)")

    similar_ranges = find_similar_segments(signal, sr, anomaly_features, threshold = 1)
    print(similar_ranges)

    for idx, (start, end) in enumerate(similar_ranges):
        t_start, t_end = samples_to_time(start, end, sr)
        print(f"Match {idx + 1}: {t_start}s - {t_end}s")

    # for idx, (start, end) in enumerate(similar_ranges):
    #     match_clip = signal[start:end]
    #     plot_waveform(match_clip, sr, title = f"Match {idx + 1}")

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    output_path = os.path.join(args.output_dir, f"{base_name}_Modified.wav")

    
    output_audio = AudioSegment(signal.tobytes(), frame_rate = sr, sample_width = 2, channels = 1)
    output_audio.export(output_path, format="wav")
    print(f"[INFO] Saved Placeholder output to: {output_path}")

if __name__ == "__main__":
    main()