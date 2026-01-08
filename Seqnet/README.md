# Seqnet — LSTM Text Generator

Seqnet is a deep learning–based text generation project that predicts the **next word in a sequence** using an **LSTM neural network** built with TensorFlow and Keras.

Given a seed text, the model generates coherent text word-by-word based on learned language patterns.

---

## Features

- LSTM-based sequence modeling
- Predicts next words using softmax probabilities
- Supports custom training text
- Generates text from a user-provided seed
- Save & load trained models

---

## Model Architecture

- **Embedding Layer** (100 dimensions)
- **LSTM Layer** (150 units, return sequences)
- **Dropout** (0.2)
- **LSTM Layer** (100 units)
- **Dense Output Layer** (Softmax over vocabulary)


