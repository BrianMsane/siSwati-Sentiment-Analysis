"""
Module for training Long-Shoterm Memory
"""

import os
import logging
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score
)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D, Bidirectional # type: ignore
from keras_tuner import HyperModel, Objective # type: ignore
from keras_tuner.tuners import RandomSearch # type: ignore
from models import load_data, clean_data, save_load_model
from lstm import tokenize


warnings.filterwarnings("ignore")
np.random.seed(42)
logging.basicConfig(level=logging.INFO, filename='models.log', filemode='a')


class BiLSTMHyperModel(HyperModel):

    def __init__(self, max_features, input_length):
        self.max_features = max_features
        self.input_length = input_length


    def build(self, hp):
        model = Sequential()
        embedding_dim = hp.Int('embedding_dim', min_value=64, max_value=256, step=32)
        model.add(Embedding(self.max_features, embedding_dim, input_length=self.input_length))
        model.add(SpatialDropout1D(hp.Float('spatial_dropout', min_value=0.1, max_value=0.5, step=0.1)))
        model.add(Bidirectional(LSTM(
            units=hp.Int('lstm_units', min_value=50, max_value=200, step=50),
            dropout=hp.Float('lstm_dropout', min_value=0.1, max_value=0.5, step=0.1),
            recurrent_dropout=hp.Float('recurrent_dropout', min_value=0.1, max_value=0.5, step=0.1)
        )))
        model.add(Dense(units=hp.Int('dense_units', min_value=32, max_value=256, step=32),
                        activation='relu'))
        model.add(Dropout(rate=hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
        model.add(Dense(3, activation='softmax'))

        model.compile(optimizer=tf.keras.optimizers.Adam(
                    hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        return model


def my_tuner(max_features, input_length):
    tuner = RandomSearch(
        BiLSTMHyperModel(max_features, input_length),
        objective=Objective("val_accuracy", direction="max"),
        max_trials=20,
        executions_per_trial=2,
        directory='bilstm_tuning',
        project_name='bi_lstm_sentiment_analysis'
    )
    return tuner


def tuner_search(tuner, X_train, X_test, y_train, y_test ):
    tuner.search(X_train, y_train,
                epochs=10,
                validation_data=(X_test, y_test),
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                           tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5)])
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    logging.info("The best parameters are {}".format(best_hyperparameters.values))
    return best_model


def fit_bet_model(best_model, X_train, X_test, y_train, y_test):
    history = best_model.fit(X_train, y_train,
                            epochs=20,
                            batch_size=64,
                            validation_data=(X_test, y_test),
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                           tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5)])

    preds = (best_model.predict(X_test))
    preds = np.argmax(preds, axis=1)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    prec = precision_score(y_test, preds, average='weighted')
    balanced = balanced_accuracy_score(y_test, preds)
    rec = recall_score(y_test, preds, average='weighted')

    logging.info(f"Accuracy         : {acc}")
    logging.info(f"F1 Score         : {f1}")
    logging.info(f"Precision        : {prec}")
    logging.info(f"Balanced Accuracy: {balanced}")
    logging.info(f"Recall           : {rec}")


def run_tuning():
    data = load_data()
    data = clean_data(data)
    X = data['Comments']
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, stratify=y, random_state=42
    )
    X_train, tokenizer = tokenize(X_train)
    X_test, _ = tokenize(X_test, tokenizer)

    max_features, input_length = len(tokenizer.word_index) + 1, X_train.shape[1]
    tuner = my_tuner(max_features, input_length)
    best_model=tuner_search(tuner, X_train, X_test, y_train, y_test)
    save_load_model("./weights/tuned_bilstm.sav", model=best_model, save=True)
    fit_bet_model(best_model, X_train, X_test, y_train, y_test)
