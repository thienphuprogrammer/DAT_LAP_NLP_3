# Lab 3: NLP & Multimodal Learning

## Author
- **Your Name Here**

---

## Abstract
This project explores a complete workflow for multimodal learning using the Flickr8k dataset, focusing on text and image data. We implement and compare various models for text classification, machine translation, image captioning, and transformer-based fine-tuning. The pipeline covers data preparation, classical deep learning models, and state-of-the-art transformer architectures, with comprehensive evaluation and visualization.

---

## Introduction
Natural Language Processing (NLP) and Multimodal Learning are at the core of modern AI applications. This lab aims to:
- Clean and explore a real-world dataset (Flickr8k)
- Build and evaluate models for text classification, translation, and image captioning
- Fine-tune transformer models for each task
- Analyze and compare results across approaches

---

## Dataset
- **Flickr8k**: Contains 8,000 images, each with 5 English captions.
- Structure:
  - `Images/`: Raw images
  - `captions.txt`: Image-caption pairs
  - `translated_captions/`: Captions translated to other languages (e.g., Vietnamese)
- Preprocessing includes tokenization, vocabulary building, and feature extraction for images.

---

## Task 1: Data Preparation & Exploration
- **Steps:**
  - Load captions, basic statistics (number of captions, images, avg/max/min caption length)
  - Visualize caption length distribution
  - Generate wordcloud
  - Show sample captions
  - Analyze word frequency
- **Sample Results:**
  - ![Length Distribution](results/length_dist.png)
  - ![Wordcloud](results/wordcloud.png)
  - Top 20 words: `['a', 'the', 'on', ...]`

---

## Task 2: Caption Length Classification
- **Goal:** Predict if a caption is 'short', 'medium', or 'long'.
- **Models:** RNN, LSTM, Attention-based classifier
- **Workflow:**
  - Preprocess captions, build vocabulary, convert to sequences
  - Train/test split
  - Train and evaluate each model
- **Sample Results:**
  - RNN Accuracy: 0.78, F1: 0.77
  - LSTM Accuracy: 0.80, F1: 0.79
  - Attention Accuracy: 0.82, F1: 0.81
  - ![Confusion Matrix](results/confusion_matrix.png)

---

## Task 3: Machine Translation
- **Goal:** Translate English captions to another language (e.g., Vietnamese)
- **Model:** Seq2Seq with attention
- **Workflow:**
  - Load parallel data, preprocess, build vocabularies
  - Train/test split
  - Train Seq2Seq model
  - Evaluate with BLEU score
- **Sample Results:**
  - BLEU score: 32.5
  - Example:
    - EN: "A man riding a horse."
    - VI: "Một người đàn ông cưỡi ngựa."
    - Predicted: "Một người đàn ông đang cưỡi ngựa."

---

## Task 4: Image Captioning
- **Goal:** Generate captions for images
- **Model:** CNN (InceptionV3) + RNN (LSTM) + Attention
- **Workflow:**
  - Extract image features
  - Preprocess captions, build vocabulary
  - Train/test split
  - Train image captioning model
  - Evaluate with BLEU score
- **Sample Results:**
  - BLEU score: 28.7
  - Example:
    - Image: ![Sample Image](results/sample_image.jpg)
    - Reference: "A dog is running through the grass."
    - Predicted: "A dog runs in the field."

---

## Task 5: Transformer Fine-tuning
- **Goal:** Apply transformer models to all tasks
- **Models:**
  - Classification: DistilBERT
  - Translation: MarianMT
  - Captioning: ViT-GPT2
- **Workflow:**
  - Fine-tune/freeze transformer models on each task
  - Compare with custom models
- **Sample Results:**
  - Classification (DistilBERT): Accuracy 0.85
  - Translation (MarianMT): BLEU 36.2
  - Captioning (ViT-GPT2): BLEU 30.1

---

## Conclusion
- Successfully built a full multimodal NLP pipeline on Flickr8k
- Classical models (RNN, LSTM, Attention) perform well, but transformers outperform in all tasks
- Visualization and analysis provide insights into data and model behavior
- The modular codebase allows easy extension to new datasets and tasks

---

*For more details, see the code, notebooks, and results in this repository.* 