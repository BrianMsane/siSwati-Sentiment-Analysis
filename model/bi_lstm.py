"""
Module for training Bi-Directional Long-Shorterm Memory
"""

import os
import logging
import time
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    balanced_accuracy_score,
    recall_score,
)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout  # type: ignore
from models import load_data, clean_data, save_load_model
from lstm import tokenize


logging.basicConfig(level=logging.INFO, filename='models.log', filemode='a')
warnings.filterwarnings("ignore")
np.random.seed(42)


def model_fn(vocab_size, max_seq_length):
    try:
        embedding_dim = 100
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_length))
        model.add(Bidirectional(LSTM(128)))
        model.add(Dropout(0.5))
        model.add(Dense(3, activation='softmax'))

        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer='adam', metrics=['accuracy']
        )
        return model
    except Exception as e:
        logging.error("Unable to initialize the model! %s", e)
        return None


def train(model, X_train, y_train, X_test, y_test):
    try:
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5)
        start = time.time()
        model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=64,
            validation_data=(X_test, y_test),
            callbacks=[early_stop, lr_scheduler]
        )
        total = time.time() - start
        save_load_model("./weights/baseline_bilstm.sav", model=model, save=True)

        preds = model.predict(X_test)
        preds = np.argmax(preds, axis=1)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        prec = precision_score(y_test, preds, average='weighted')
        balanced = balanced_accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds, average='weighted')

        logging.info("The total amount of time it took to train Bi-LSTM is {}".format(total))
        logging.info(f"Accuracy         : {acc}")
        logging.info(f"F1 Score         : {f1}")
        logging.info(f"Precision        : {prec}")
        logging.info(f"Balanced Accuracy: {balanced}")
        logging.info(f"Recall           : {rec}")
        return True
    except Exception as e:
        logging.error("Unable to train and evaluate! %s", e)
        return False


def main():
    try:
        path = '../data/'
        name = 'Siswati_Sentiment.csv'
        data = load_data(name)
        data = clean_data(data)

        X = data['Comments']
        y = data['label']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.2, stratify=y, random_state=42
        )
        X_train, tokenizer = tokenize(x=X_train)
        X_test, _ = tokenize(x=X_test)
        max_seq_length = X_train.shape[1]
        vocab_size = len(tokenizer.word_index) + 1

        model = model_fn(vocab_size=vocab_size, max_seq_length=max_seq_length)
        train(model=model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        return True
    except Exception as e:
        logging.error("Unable to run the main function! %s", e)
        return False
