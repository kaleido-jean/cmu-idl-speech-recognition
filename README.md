
# Phoneme Recognition from MFCC Frames (PyTorch)

A compact speech DL project that performs **frame-level phoneme classification** using **MFCC features + contextual windowing**, trained in **PyTorch** with optional **SpecAugment (time/frequency masking)** and experiment tracking via **Weights & Biases**.



---

## Highlights

- **End-to-end pipeline**: data loading → MFCC normalization → context-window features → training/validation → test inference; integrates **wandb** logging
- **Context modeling**: each training sample uses a window of \(2C+1\) frames to predict the center-frame phoneme
- **Augmentation**: **FrequencyMasking + TimeMasking** (applied with probability)
- **Efficient training**: GPU support + **mixed precision (AMP)** + configurable batch size

---

## Task

Given MFCC features extracted from speech audio, predict a phoneme label for each time frame.

- Input: MFCC frame features (28 dims per frame) with context window
- Output: one of the phoneme classes (40 in total)

---

## Project Structure (Notebook)

This project is implemented as a notebook/script style workflow:

1. Install dependencies (`wandb`, `torchaudio`)
2. Download/unzip dataset (Kaggle competition format)
3. Define phoneme set
4. Build datasets + dataloaders
5. Define model (MLP)
6. Train/evaluate (CE loss, optimizer + scheduler)
7. Run inference on test set
8. Export `submission.csv`

---

## Method Overview

### 1) Data Processing
- Loads per-utterance MFCC numpy arrays and transcripts
- Applies **cepstral normalization** over time:
  - standardization per feature dimension (mean/std computed along time axis)
- Concatenates all utterances into a single long sequence for frame sampling
- Pads MFCCs with zeros to support context frames at boundaries

### 2) Context Window Features
For each index `ind`, the sample is:
- `frames = mfcc[ind : ind + 2*context + 1]`  → shape \((2C+1, 28)\)
- label is the phoneme at `ind`

Model input is flattened to:
- `INPUT_SIZE = (2*context + 1) * 28`

### 3) Model
A simple MLP network in pyramid architecture:

- Block: Linear + BatchNorm1d + GeLU + Dropout
- Params: 13M

### 4) Training Setup
- Loss: `CrossEntropyLoss`
- Optimizer: `AdamW`
- Scheduler: `ReduceLROnPlateau`
- Mixed precision training: `torch.autocast` + `GradScaler`

### 5) Augmentation
Applied in the training `collate_fn` with probability:
- `FrequencyMasking`
- `TimeMasking`

---

## Configuration

| Hyperparameters | Values |
|---|---|
| Number of Layers | 2–8 |
| Activations | ReLU, LeakyReLU, softplus, tanh, sigmoid |
| Batch Size | 64, 128, 256, 512, 1024, 2048 |
| Architecture | Cylinder, Pyramid, Inverse-Pyramid, Diamond |
| Dropout | 0–0.5, Dropout in alternate layers |
| LR Scheduler | Fixed, StepLR, ReduceLROnPlateau, Exponential, CosineAnnealing |
| Weight Initialization | Gaussian, Xavier, Kaiming (Normal and Uniform), Random, Uniform |
| Context | 0–50 |
| Batch-Norm | Before or After Activation, Every layer or Alternate Layer or No Layer |
| Optimizer | Vanilla SGD, Nesterov’s momentum, RMSProp, Adam |
| Augmentation | Frequent Masking, Time Masking |
| Regularization | Weight Decay |
| LR | 0.001, you can experiment with this |
| Normalization | You can try Cepstral Normalization |

---

## Output

- `submission.csv` with format:
  - header: `id,label`
  - each row: frame index and predicted phoneme string

---

## What I Learned / What This Demonstrates (for Applications)

- Building a **clean MLP training pipeline**: datasets, dataloaders, normalization, batching, inference
- Implementing **context-window modeling** for sequential audio features
- Using **frequency augmentations** and **mask augmentation** to improve robustness
- Tracking experiments with **wandb** and organizing hyperparameters for reproducibility

---

## Acknowledgements
The original starter codes and writeup remain the intellectual property of the CMU Introduction to Deep Learning course professor and teaching assistants and are subject to the course’s copyright and licensing terms. Please do not use or distribute any part of this repository in ways that violate course policies or academic integrity guidelines.
