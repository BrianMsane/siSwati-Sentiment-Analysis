
import time
import logging
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from skopt import BayesSearchCV
from models import load_data, clean_data, split_data, evaluate_model, save_load_model
logging.basicConfig(level=logging.INFO, filename='models.log', filemode='a')


def fine_tune_svc(X_train, X_test, y_train, y_test):
    search_space = {
        'C': (1e-3, 100.0, 'log-uniform'),
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
    tuner = BayesSearchCV(
        SVC(),
        search_space,
        scoring='accuracy',
        n_iter=10,
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    tuner.fit(X_train, y_train)
    best_params = tuner.best_params_
    best_model = tuner.best_estimator_

    start = time.time()
    best_model.fit(X_train, y_train)
    total = time.time() - start
    logging.info("The time it took to trained the best (tuned) SVM is {}".format(total))
    save_load_model("./weights/tuned_svc.sav", model=best_model, save=True)

    preds = best_model.predict(X_test)
    logging.info("These are the best scores for the tuned SVM model")
    evaluate_model(preds=preds, y_test=y_test)
    logging.info(f"SVM Best hyperparameters: {best_params}")


def fine_tune_xgboost(X_train, X_test, y_train, y_test):
    search_space = {
        'n_estimators': (50, 500),
        'max_depth': (3, 20),
        'learning_rate': (1e-4, 1e-1, 'log-uniform'),
        'subsample': (0.5, 1.0)
    }
    tuner = BayesSearchCV(
        xgb.XGBClassifier(),
        search_space,
        scoring='accuracy',
        n_iter=10,
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    tuner.fit(X_train, y_train)
    best_params = tuner.best_params_
    best_model = tuner.best_estimator_
    start = time.time()
    best_model.fit(X_train, y_train)
    total = time.time() - start
    logging.info("The time it took to trained the best (tuned) XGBoost is {}".format(total))
    save_load_model("./weights/tuned_xgboost.sav", model=best_model, save=True)

    preds = best_model.predict(X_test)
    logging.info("These are the best scores for the tuned XGBoost model")
    evaluate_model(preds=preds, y_test=y_test)
    logging.info(f"XGBoost Best hyperparameters: {best_params}")


def fine_tune_adaboost(X_train, X_test, y_train, y_test):
    search_space = {
        'estimator__max_depth': (1, 10),
        'n_estimators': (50, 500),
        'learning_rate': (1e-4, 1e-1, 'log-uniform')
    }
    tuner = BayesSearchCV(
        AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', splitter='random')),
        search_space,
        scoring='accuracy',
        n_iter=10,
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    tuner.fit(X_train, y_train)
    best_params = tuner.best_params_
    best_model = tuner.best_estimator_
    start = time.time()
    best_model.fit(X_train, y_train)
    total = time.time() - start
    logging.info("The time it took to trained the best (tuned) AdaBoost is {}".format(total))
    save_load_model("./weights/tuned_adaboost.sav", model=best_model, save=True)

    preds = best_model.predict(X_test)
    logging.info("These are the best scores for the tuned AdaBoost model")
    evaluate_model(preds=preds, y_test=y_test)
    logging.info(f"AdaBoost Best hyperparameters: {best_params}")


def fine_tune_naive_bayes(X_train, X_test, y_train, y_test):
    search_space = {
        'alpha': (1e-3, 1.0, 'log-uniform')
    }
    tuner = BayesSearchCV(
        MultinomialNB(),
        search_space,
        scoring='accuracy',
        n_iter=10,
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    tuner.fit(X_train, y_train)
    best_params = tuner.best_params_
    best_model = tuner.best_estimator_
    start = time.time()
    best_model.fit(X_train, y_train)
    total = time.time() - start
    logging.info("The time it took to trained the best (tuned) Naive Bayes is {}".format(total))
    save_load_model("./weights/tuned_bayes.sav", model=best_model, save=True)

    preds = best_model.predict(X_test)
    logging.info("These are the best scores for the tuned Naive Bayes model")
    evaluate_model(preds=preds, y_test=y_test)
    logging.info(f"Naive Bayes Best hyperparameters: {best_params}")


def run_tuning():
    data = load_data()
    data = clean_data(data)

    X_train, X_test, y_train, y_test = split_data(data=data)
    fine_tune_svc(X_train, X_test, y_train, y_test)
    fine_tune_xgboost(X_train, X_test, y_train, y_test)
    fine_tune_adaboost(X_train, X_test, y_train, y_test)
    fine_tune_naive_bayes(X_train, X_test, y_train, y_test)
