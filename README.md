# DINOv3-LungSound

This repository contains the implementation used for the final project:

**DINOv3 Representation Analysis for Respiratory Sound Spectrograms: Limitations of Natural Image Foundation Models for Lung Sound Classification**

The project evaluates whether a self-supervised vision foundation model pretrained on natural images, DINOv3, can effectively represent and classify respiratory sound spectrograms.

## Overview

The experimental pipeline consists of two stages.

### Stage 1: Representation Analysis

DINOv3 embeddings are extracted from respiratory sound spectrogram images and analyzed using:

- Silhouette score
- k-nearest neighbor classification
- t-SNE visualization
- UMAP visualization
- Class centroid distance analysis

### Stage 2: Classification Evaluation

Several downstream classifiers are evaluated:

- DINOv3 CLS embedding + Logistic Regression
- DINOv3 CLS embedding + MLP
- DINOv3 fine-tuning
- DINOv3 fine-tuning with focal loss
- VGG11-BN baseline
- ResNet18 baseline

## Dataset

This project uses the ICBHI Respiratory Sound Database.

The processed metadata file should include at least the following columns:

- `image_path`
- `label`
- `label_id`
- `split`
- `patient_id`

Expected class labels:

- `normal`
- `crackle`
- `wheeze`
- `both`

## Expected Directory Structure

Before running the scripts, arrange the dataset as follows:

```text
ICBHI_final_database/
├── dinov3_patient_split_metadata.csv
├── logmel_delta_deltadelta_3ch_224/
│   ├── normal/
│   ├── crackle/
│   ├── wheeze/
│   └── both/
└── outputs/
```

The image files should be 224×224 three-channel spectrogram images generated from log-mel, delta, and delta-delta features.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Extract DINOv3 embeddings

```bash
python stage1_representation/extract_dinov3_embeddings.py \
  --root "C:/path/to/ICBHI_final_database"
```

### 2. Analyze embeddings

```bash
python stage1_representation/analyze_embeddings.py \
  --root "C:/path/to/ICBHI_final_database"
```

### 3. Train linear and MLP probes

```bash
python stage2_classification/linear_mlp_probe.py \
  --root "C:/path/to/ICBHI_final_database"
```

### 4. Fine-tune DINOv3

```bash
python stage2_classification/dinov3_finetune.py \
  --root "C:/path/to/ICBHI_final_database"
```

### 5. Fine-tune DINOv3 with focal loss

```bash
python stage2_classification/dinov3_focal_finetune.py \
  --root "C:/path/to/ICBHI_final_database"
```

### 6. Train CNN baselines

```bash
python stage2_classification/vgg11_baseline.py \
  --root "C:/path/to/ICBHI_final_database"

python stage2_classification/resnet18_baseline.py \
  --root "C:/path/to/ICBHI_final_database"
```

## Code Availability Statement

The source code used in this project is publicly available at GitHub:

```text
https://github.com/your-github-id/DINOv3-LungSound
```

The repository includes preprocessing utilities, DINOv3 embedding extraction, representation analysis, downstream classification models, CNN baselines, and evaluation scripts required to reproduce the main experimental results.

## Main Experimental Findings

The representation analysis showed that DINOv3 embeddings did not naturally form well-separated respiratory sound clusters. The silhouette score was negative, and t-SNE/UMAP visualizations showed substantial class overlap. Although fine-tuning improved downstream performance, the results suggest that the pretrained natural-image representation itself is not sufficiently adapted to fine-grained respiratory sound spectrogram patterns, especially transient crackle events.

## Notes

- Update all paths according to your local environment.
- The ICBHI dataset itself is not included in this repository.
- Large model checkpoints and generated outputs are ignored by `.gitignore`.
