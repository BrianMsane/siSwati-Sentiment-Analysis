"""
Module for training and evaluating Machine Learning models.
"""

import os
import logging
import warnings
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, recall_score, precision_score
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
# from sswcleaner import TextPreprocessor # type: ignore
# from utils.clean import DatasetPreprocessor, TextPreprocessor
import xgboost as xgb # type: ignore

warnings.filterwarnings("ignore")
np.random.seed(42)
logging.basicConfig(level=logging.INFO, filemode='a', filename='models.log')


def load_data(path: str='Siswati_Sentiment.csv') -> pd.DataFrame:
    """
    Load the dataset from the given path.
    """
    try:
        extension = os.path.splitext(path)[1]
        if extension == '.xlsx':
            dataframe = pd.read_excel(path)
            return dataframe
        elif extension == '.csv':
            dataframe = pd.read_csv(path, encoding='latin')
            return dataframe
    except Exception as e:
        logging.error("Unable to load data: %s", str(e))
        return None


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset before training.
    """
    try:
        # cleaner = DatasetPreprocessor()
        # data = cleaner.clean_dataset(data)
        # preprocessor = TextPreprocessor()
        # data['Comments'] = data['Comments'].apply(preprocessor.clean_text)
        data['label'] = data['label'].replace(-1, 2)
        return data
    except Exception as e:
        logging.error("Unable to clean data: %s", str(e))
        return None


def transform_data(x: pd.Series):
    """
    Apply TF-IDF vectorization to the data.
    """
    try:
        vectorizer = TfidfVectorizer()
        x = vectorizer.fit_transform(x)
        return x
    except Exception as e:
        logging.error("Unable to transform data: %s", str(e))
        return np.array([])


def split_data(data: pd.DataFrame):
    """
    Split the data into training and testing sets.
    """
    try:
        X = data['Comments']
        y = data['label']
        X = transform_data(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.2, stratify=y, random_state=42
        )
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error("Unable to split data: %s", str(e))
        return None


def handle_imbalance(X_train, y_train):
    """
    Handle class imbalance using SMOTE.
    """
    try:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        return X_train, y_train
    except Exception as e:
        logging.error("Unable to handle imbalance: %s", str(e))
        return None


def train_model(model, data: pd.DataFrame):
    """
    Train the model and make predictions.
    """
    try:
        X_train, X_test, y_train, y_test = split_data(data)
        # X_train, y_train = handle_imbalance(X_train, y_train)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return preds, y_test
    except Exception as e:
        logging.error("Unable to train model: %s", str(e))
        return None


def evaluate_model(preds, y_test):
    """
    Evaluate the model's performance.
    """
    try:
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
        return True
    except Exception as e:
        logging.error("Unable to evaluate model: %s", str(e))
        return False


def save_load_model(path: str, model=None, save=False, load=False):
    """
    Save or load a model.
    """
    try:
        if save:
            joblib.dump(model, path)
        if load:
            return joblib.load(path)
    except Exception as e:
        logging.error("Unable to save/load model: %s", str(e))
        return None


def training_pipeline(models: list[str], data: pd.DataFrame):
    """
    Train, evaluate, and save models.
    """
    try:
        for model_name in models:
            if model_name == 'SupportVectorMachines':
                model = SVC(random_state=42, C=50, gamma=.01, kernel='rbf')
            elif model_name == 'XGBoost':
                model = xgb.XGBClassifier(
                    random_state=42, n_estimators=300, learning_rate=.1,
                    subsample=1, colsample_bytree=1, gamma=0, max_depth=9
                )
            elif model_name == 'NaiveBayes':
                model = MultinomialNB()
            elif model_name == 'AdaBoost':
                model = AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(criterion='entropy', splitter='random'),
                    n_estimators=50, random_state=42
                )

            start = time.time()
            preds, y_test = train_model(model, data)
            total = time.time() - start
            logging.info("The total time it took to train the {} is {}".format(model_name, total))

            save_load_model(f"./weights/{model_name}.sav", model=model, save=True)
            evaluate_model(preds, y_test)
        return True

    except Exception as e:
        logging.error("Unable to run training pipeline: %s", str(e))
        return False


def main():
    """
    Execute the training and prediction pipelines.
    """
    try:
        # path = "../data/"
        name = 'Siswati_Sentiment.csv'
        data = load_data(name)
        data = clean_data(data)

        models = [
            "SupportVectorMachines",
            "XGBoost",
            "NaiveBayes",
            "AdaBoost"
        ]
        done = training_pipeline(models, data)
        if done:
            return True
        return False

    except Exception as e:
        logging.error("Unable to execute main function: %s", str(e))
        return False
