# DINOv3-LungSound

## Overview

This repository contains the implementation used for the final project:

**DINOv3 Representation Analysis for Respiratory Sound Spectrograms: Limitations of Natural Image Foundation Models for Lung Sound Classification**

The objective of this project is to investigate whether DINOv3, a self-supervised vision foundation model pretrained on natural images, can effectively represent and classify respiratory sound spectrograms. The study consists of two stages: (1) representation analysis of frozen DINOv3 embeddings and (2) downstream classification evaluation using DINOv3 and CNN-based models.

---

## Dataset

Experiments were conducted using the **ICBHI Respiratory Sound Database**.

The dataset contains 6,898 respiratory cycles:

| Class   | Samples |
| ------- | ------: |
| Normal  |   3,642 |
| Crackle |   1,864 |
| Wheeze  |     886 |
| Both    |     506 |
| Total   |   6,898 |

A patient-level split was applied to prevent information leakage:

| Split      | Samples |
| ---------- | ------: |
| Train      |   3,502 |
| Validation |   1,587 |
| Test       |   1,809 |

---

## Repository Structure

```text
DINOv3-LungSound/
│
├── data_preprocessing/
│   ├── generate_spectrograms.py
│   └── patient_split.py
│
├── stage1_representation/
│   ├── extract_dinov3_embeddings.py
│   ├── analyze_embeddings.py
│   └── visualize_tsne_umap.py
│
├── stage2_classification/
│   ├── linear_mlp_probe.py
│   ├── dinov3_finetune.py
│   └── cnn_audio_concat_baseline.py
│
├── utils.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## File Descriptions

### data_preprocessing

**generate_spectrograms.py**

Generates 224×224 three-channel spectrogram images using log-mel, delta, and delta-delta features from respiratory sound recordings.

**patient_split.py**

Creates patient-level train, validation, and test splits to prevent patient overlap across datasets.

---

### stage1_representation

**extract_dinov3_embeddings.py**

Extracts CLS-token and mean-pooled embeddings from DINOv3-ViT-L/16.

**analyze_embeddings.py**

Computes representation quality metrics including silhouette score, k-NN accuracy, and class centroid distances.

**visualize_tsne_umap.py**

Generates t-SNE and UMAP visualizations for qualitative embedding analysis.

---

### stage2_classification

**linear_mlp_probe.py**

Evaluates frozen DINOv3 embeddings using Logistic Regression and MLP classifiers.

**dinov3_finetune.py**

Fine-tunes DINOv3 using Cross-Entropy Loss or Focal Loss for respiratory sound classification.

**cnn_audio_concat_baseline.py**

Trains CNN baseline models (VGG11-BN, VGG16, ResNet18, ResNet34) with handcrafted audio feature fusion.

---

### Root Files

**utils.py**

Utility functions used across preprocessing, representation analysis, and classification experiments.

**requirements.txt**

Required Python packages for reproducing the experiments.





