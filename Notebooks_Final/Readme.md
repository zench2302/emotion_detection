# Speech Emotion Recognition with Deep Learning

This repository contains notebooks for a Speech Emotion Recognition (SER) project. It explores various audio feature extraction techniques—ranging from classical signal processing (MFCC, Log Mel Spectrogram) to state-of-the-art self-supervised models (HuBERT, Wav2Vec2)—and evaluates their performance on a multi-corpus emotion classification task.

## Contents

### 1. `Final Data Preprocessing.ipynb`
- Downloads and prepares four emotional speech datasets from Kaggle: RAVDESS, TESS, SAVEE, and CREMA-D
- Harmonizes label formats and audio metadata
- Performs Exploratory Data Analysis (EDA), including:
  - Distribution of audio duration
  - Emotion label counts
  - Sampling rate consistency
  - Box plots: duration vs. emotion
  - Box plots: RMS energy vs. emotion

---

### 2. `Final MFCC.ipynb`
- Extracts MFCCs (40 coefficients) + energy feature from each audio file
- Trains various neural network architectures:
  - Deep CNN (4 Conv blocks)
  - CNN-BiLSTM with Attention

---

### 3. `Final Mel Spectrogram.ipynb`
- Extracts Log-Mel Spectrograms 
- Best Mel Spectrogram model
- Final model shares architecture with MFCC1
- Reports classification performance

---

### 4. `Final Hubert.ipynb`
- Uses pre-trained HuBERT to extract 768-dimensional frame-wise embeddings
- Applies temporal pooling (mean + std) to reduce features to 1536-dim vectors
- Trains simple CNN + RNN models

---

### 5. `Final Wav2Vec2.ipynb`
- Applies Wav2Vec2.0 for frame-level feature extraction
- Trains models on frozen and partially fine-tuned representations
- Best-performing model across all feature sets (accuracy: 79.41%)
- Performs model sanity checks using __

---

### 6. `XAI Mel MFCC.ipynb`
- Performs model sanity checks using Grad-CAM
- Visualizes heatmaps over MFCC and Log-Mel inputs for different emotions
- Compares feature interpretability across feature types

---
### 6. `XAI HuBERT.ipynb`
- Performs model sanity checks using __
- Visualizes heatmaps and boxplots over HuBERT for different emotions
