import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import os

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

def generate_text(seed_text, next_words, model, max_sequence_len, tokenizer):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        predicted = np.argmax(predicted_probs, axis=-1)
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        
        seed_text += " " + output_word
    return seed_text.title()

def save_model(model, path):
    model.save(path)

def load_saved_model(path):
    if os.path.exists(path):
        return tf.keras.models.load_model(path)
    return None
