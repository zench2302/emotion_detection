## Project proposal form

Please provide the information requested in the following form. Try provide concise and informative answers.

**1. What is your project title?**

Emotion Recognition from Speech using Deep Learning models.

**2. What is the problem that you want to solve?**

To develop a system that can recognize human emotions from speech using deep learning methods, aiming to understand how emotional states manifest in acoustic features and evaluate if modern models like HuBERT outperform traditional approaches.

**3. What deep learning methodologies do you plan to use in your project?**
Self-supervised learning: HuBERT, Wav2Vec2 for feature extraction
Sequence models: LSTM / Bi-LSTM for time-dependent features
Transformer-based models for better long-term context modeling
Optionally: simple CNN on spectrograms

**4. What dataset will you use? Provide information about the dataset, and a URL for the dataset if available. Briefly discuss suitability of the dataset for your problem.**

[Dataset](https://www.kaggle.com/code/ejlok1/audio-emotion-part-1-explore-data?scriptVersionId=20844990)
- 11106 labeled recordings with 8 different emotions, including male and female.
- Each sample includes high-quality audio with emotion labels (e.g., happy, sad, angry, neutral).
- Suitable for supervised training and evaluation of classification models.

**5. List key references (e.g. research papers) that your project will be based on?**

[HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction](https://arxiv.org/pdf/2106.07447)
Introduces HuBERT model which can be used for extracting speech features without labeled data.

[Speech Emotion Recognition: Features and Classification Models](https://arxiv.org/abs/2001.11409)
A comprehensive review of traditional and deep learning methods used for emotion recognition from speech.

[Learning Representations for Speech Emotion Recognition using Self-Supervised Models](https://arxiv.org/pdf/2104.03502.pdf)
Explores how self-supervised models like Wav2Vec2 can outperform handcrafted features in emotion recognition tasks.

[Baseline Model](https://www.kaggle.com/code/ejlok1/audio-emotion-part-3-baseline-model#4.-Model-validation)

**Please indicate whether your project proposal is ready for review (Yes/No):**
Yes
## Feedback (to be provided by the course lecturer)

[MV] 27 March 2025. **Approved**.

Sounds interesting. The proposed topic would allow you to explore a range of deep learning methods, as outlined in the proposal.
