"""
Module for training Long-Shoterm Memory
"""

import os
import logging
import warnings
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score
)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from models import load_data, clean_data, save_load_model

warnings.filterwarnings("ignore")
np.random.seed(42)
logging.basicConfig(level=logging.INFO, filename='models.log', filemode='a')


def tokenize(x, max_features=10000, maxlen=100, min_freq=5):
    try:
        tokenizer = Tokenizer()#num_words=max_features)
        tokenizer.fit_on_texts(x)
        x = tokenizer.texts_to_sequences(x)
        x = pad_sequences(x, maxlen=maxlen)#, padding='post', truncating='post')
        return x, tokenizer
    except Exception as e:
        logging.error("Unable to tokenize the data! %s", e)
        return None


def model_df(X_train, max_features=10000):
    try:
        embedding_dim = 128
        model = Sequential()
        model.add(Embedding(max_features, embedding_dim, input_length=X_train.shape[1]))
        model.add(SpatialDropout1D(.2))
        model.add(LSTM(100, dropout=.2, recurrent_dropout=.2))
        model.add(Dense(3, activation='softmax'))
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer='adam', metrics=['accuracy']
        )
        return model
    except Exception as e:
        logging.error("Unable to define the model structure! %s", e)
        return None


def model_train(model, X_train, y_train, X_test, y_test):
    try:
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
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
        save_load_model("./weights/baseline_lstm.sav", model=model, save=True)

        preds = (model.predict(X_test))
        preds = np.argmax(preds, axis=1)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        prec = precision_score(y_test, preds, average='weighted')
        balanced = balanced_accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds, average='weighted')

        logging.info("The total amount of time it took to train LSTM is {}".format(total))
        logging.info(f"Accuracy         : {acc}")
        logging.info(f"F1 Score         : {f1}")
        logging.info(f"Precision        : {prec}")
        logging.info(f"Balanced Accuracy: {balanced}")
        logging.info(f"Recall           : {rec}")
        return True
    except Exception as e:
        logging.error("Exception occurred while training and evaluating! %s", e)
        return False


def main():
    try:
        path = '../data/'
        data = load_data("Siswati_Sentiment.csv")
        data = clean_data(data)
        X = data['Comments']
        y = data['label']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.2, stratify=y, random_state=42
        )
        X_train, tokenizer = tokenize(X_train)
        X_test, _ = tokenize(X_test, tokenizer)

        model = model_df(X_train=X_train)
        model = model_train(
            model=model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )
        return True
    except Exception as e:
        logging.error("Unable to run the main function! %s", e)
        return False
