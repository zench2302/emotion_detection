**Description:**  
- Developed a deep learning pipeline for robust emotion classification from speech across four benchmark datasets (SAVEE, RAVDESS, CREMA-D, TESS).
- Combined feature extraction (HuBERT, spectrograms) with advanced sequence modeling to capture nuanced prosodic and spectral patterns.

**Key Technologies:**  
- Deep learning models: CNN, BiLSTM, GRU, Attention, Multi-Head Attention
- Pre-trained speech representations: HuBERT
- Optimization: Data augmentation (noise, pitch, tempo), Label smoothing, AdamW

**Highlights:**  
- Achieved 87.6% validation accuracy with BiLSTM + Attention model after augmentation and optimization.
- Implemented joint fine-tuning of HuBERT + classifier to adapt large-scale speech representations for downstream tasks.
- Demonstrated generalization across heterogeneous datasets, simulating real-world home environments.
