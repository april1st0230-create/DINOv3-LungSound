"""
Placeholder script for spectrogram generation.

In the submitted project, the spectrogram images were generated before
DINOv3 experiments. This file documents the expected preprocessing flow.

Recommended flow:
1. Load respiratory sound waveform.
2. Apply optional band-pass filtering.
3. Generate log-mel spectrogram.
4. Compute delta and delta-delta features.
5. Stack [log-mel, delta, delta-delta] as RGB channels.
6. Resize or save as 224 x 224 image.
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_audio_dir", type=str, required=False)
    parser.add_argument("--output_image_dir", type=str, required=False)
    args = parser.parse_args()

    print("This is a template placeholder.")
    print("Use your existing spectrogram generation notebook/script here.")
    print("Expected output: 224x224 RGB images grouped by class folder.")


if __name__ == "__main__":
    main()
