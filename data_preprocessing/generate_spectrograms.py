# -*- coding: utf-8 -*-
"""
Generate 3-channel respiratory sound spectrogram images.

Output RGB channels:
    R: log-mel spectrogram
    G: delta feature
    B: delta-delta feature

Expected ICBHI-style structure:
    input_audio_dir/
        101_1b1_Al_sc_Meditron.wav
        101_1b1_Al_sc_Meditron.txt
        ...

Annotation txt format:
    start_time  end_time  crackle  wheeze

Label rule:
    crackle=0, wheeze=0 -> normal
    crackle=1, wheeze=0 -> crackle
    crackle=0, wheeze=1 -> wheeze
    crackle=1, wheeze=1 -> both
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import librosa
import scipy.signal as signal


LABEL2ID = {
    "normal": 0,
    "crackle": 1,
    "wheeze": 2,
    "both": 3,
}


def bandpass_filter(y, sr, lowcut=50.0, highcut=2500.0, order=4):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)

    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, y)


def normalize_to_uint8(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.min(x)
    denom = np.max(x) + 1e-8
    x = x / denom
    x = (x * 255.0).clip(0, 255).astype(np.uint8)
    return x


def pad_or_truncate(y, target_len):
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)), mode="constant")
    return y[:target_len]


def get_label(crackle, wheeze):
    crackle = int(crackle)
    wheeze = int(wheeze)

    if crackle == 0 and wheeze == 0:
        return "normal"
    if crackle == 1 and wheeze == 0:
        return "crackle"
    if crackle == 0 and wheeze == 1:
        return "wheeze"
    return "both"


def make_logmel_delta_image(
    y,
    sr,
    duration=8.0,
    n_mels=128,
    n_fft=1024,
    hop_length=256,
    image_size=224,
):
    target_len = int(sr * duration)
    y = pad_or_truncate(y, target_len)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )

    logmel = librosa.power_to_db(mel, ref=np.max)
    delta = librosa.feature.delta(logmel)
    delta2 = librosa.feature.delta(logmel, order=2)

    logmel_u8 = normalize_to_uint8(logmel)
    delta_u8 = normalize_to_uint8(delta)
    delta2_u8 = normalize_to_uint8(delta2)

    rgb = np.stack([logmel_u8, delta_u8, delta2_u8], axis=-1)

    img = Image.fromarray(rgb)
    img = img.resize((image_size, image_size), resample=Image.BILINEAR)

    return img


def read_annotation(txt_path):
    rows = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            start = float(parts[0])
            end = float(parts[1])
            crackle = int(parts[2])
            wheeze = int(parts[3])

            rows.append((start, end, crackle, wheeze))

    return rows


def process_file(
    wav_path,
    txt_path,
    output_dir,
    target_sr=4000,
    duration=8.0,
    apply_bpf=True,
):
    basename = os.path.splitext(os.path.basename(wav_path))[0]
    annotations = read_annotation(txt_path)

    y_full, sr = librosa.load(wav_path, sr=target_sr, mono=True)

    if apply_bpf:
        y_full = bandpass_filter(y_full, sr)

    metadata_rows = []

    for idx, (start, end, crackle, wheeze) in enumerate(annotations):
        start_sample = int(start * sr)
        end_sample = int(end * sr)

        segment = y_full[start_sample:end_sample]

        label = get_label(crackle, wheeze)
        label_id = LABEL2ID[label]

        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        out_name = f"{basename}_cycle{idx:03d}_{label}.png"
        out_path = os.path.join(label_dir, out_name)

        img = make_logmel_delta_image(
            segment,
            sr=sr,
            duration=duration,
        )

        img.save(out_path)

        patient_id = basename.split("_")[0]

        metadata_rows.append({
            "patient_id": patient_id,
            "recording_id": basename,
            "cycle_index": idx,
            "start_time": start,
            "end_time": end,
            "crackle": crackle,
            "wheeze": wheeze,
            "label": label,
            "label_id": label_id,
            "image_path": out_path,
        })

    return metadata_rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_audio_dir",
        type=str,
        required=True,
        help="Directory containing ICBHI wav and txt files.",
    )
    parser.add_argument(
        "--output_image_dir",
        type=str,
        required=True,
        help="Directory where spectrogram images will be saved.",
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default="spectrogram_metadata.csv",
        help="Output metadata CSV path.",
    )
    parser.add_argument(
        "--target_sr",
        type=int,
        default=4000,
        help="Target sampling rate.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
        help="Fixed segment duration in seconds.",
    )
    parser.add_argument(
        "--no_bpf",
        action="store_true",
        help="Disable band-pass filtering.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_image_dir, exist_ok=True)

    wav_files = sorted([
        os.path.join(args.input_audio_dir, f)
        for f in os.listdir(args.input_audio_dir)
        if f.lower().endswith(".wav")
    ])

    all_metadata = []

    for wav_path in tqdm(wav_files, desc="Generating spectrograms"):
        txt_path = os.path.splitext(wav_path)[0] + ".txt"

        if not os.path.exists(txt_path):
            print(f"[WARNING] Missing annotation file: {txt_path}")
            continue

        rows = process_file(
            wav_path=wav_path,
            txt_path=txt_path,
            output_dir=args.output_image_dir,
            target_sr=args.target_sr,
            duration=args.duration,
            apply_bpf=not args.no_bpf,
        )

        all_metadata.extend(rows)

    meta_df = pd.DataFrame(all_metadata)
    meta_df.to_csv(args.metadata_csv, index=False, encoding="utf-8-sig")

    print("Done.")
    print("Saved images to:", args.output_image_dir)
    print("Saved metadata to:", args.metadata_csv)

    if len(meta_df) > 0:
        print(meta_df["label"].value_counts())
        print("Total samples:", len(meta_df))


if __name__ == "__main__":
    main()
