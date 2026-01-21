import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import json
from datetime import datetime

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def create_tokenizer(text):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    return tokenizer

def create_sequences(tokenizer, text, max_sequence_len):
    input_sequences = []
    # Split text by new lines to treat them as separate potential sequences,
    # or just simple windowing over the whole text. 
    # For better context, we'll split by lines first.
    lines = text.split('\n')
    
    for line in lines:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
            
    # Pad sequences
    max_sequence_len = max([len(x) for x in input_sequences]) if input_sequences else 10
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = tf.keras.utils.to_categorical(label, num_classes=len(tokenizer.word_index) + 1)
    
    return predictors, label, max_sequence_len

def save_tokenizer(tokenizer, path):
    with open(path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_tokenizer(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def save_dataset_metadata(metadata, path="dataset_metadata.json"):
    metadata["timestamp"] = datetime.utcnow().isoformat()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(metadata) + "\n")

def compute_dataset_stats(text):
    lines = text.splitlines()
    return {
        "lines": len(lines),
        "characters": len(text),
        "avg_line_length": sum(len(line) for line in lines) / max(len(lines), 1)
    }
