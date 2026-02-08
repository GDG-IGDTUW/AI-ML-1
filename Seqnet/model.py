import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import os

def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature  # avoid log(0)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def create_model(total_words, max_sequence_len):
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
    model.add(LSTM(128, return_sequences=True))  # reduced units
    model.add(Dropout(0.3))                     # increased dropout
    model.add(LSTM(64))                          # reduced units
    model.add(Dropout(0.2))                     # optional extra dropout
    model.add(Dense(total_words, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
    return model

def train_model(model, predictors, label, epochs=50, validation_split=0.2, patience=5):
    """
    Trains the LSTM model with validation split and early stopping to reduce overfitting.
    
    Args:
        model: Keras model to train.
        predictors: Input sequences (X).
        label: One-hot encoded labels (y).
        epochs: Maximum number of epochs.
        validation_split: Fraction of data to use as validation.
        patience: Number of epochs with no improvement to stop training early.
    
    Returns:
        Trained Keras model and training history.
    """
    
    early_stop = EarlyStopping(
        monitor='val_loss',        # monitor validation loss
        patience=patience,         # stop after 'patience' epochs with no improvement
        restore_best_weights=True  # restore best weights from training
    )
    
    history = model.fit(
        predictors,
        label,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=1
    )
    
    return model, history

def generate_text(seed_text, next_words, model, max_sequence_len, tokenizer, temperature=1.0, repetition_threshold=2):
    recent_words = []

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = tf.keras.preprocessing.sequence.pad_sequences(
            [token_list], maxlen=max_sequence_len-1, padding='pre'
        )
        predicted_probs = model.predict(token_list, verbose=0)[0]

        # Penalize recently used words
        for word_idx in recent_words:
            if word_idx < len(predicted_probs):
                predicted_probs[word_idx] *= 0.1  # reduce probability

        predicted_index = sample_with_temperature(predicted_probs, temperature)

        # Map index back to word
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break

        # Update recent words
        recent_words.append(predicted_index)
        if len(recent_words) > repetition_threshold:
            recent_words.pop(0)

        seed_text += " " + output_word

    return seed_text.title()


def save_model(model, path):
    model.save(path)

def load_saved_model(path):
    if os.path.exists(path):
        return tf.keras.models.load_model(path)
    return None
