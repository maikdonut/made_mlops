import pickle
import numpy as np
from typing import Dict, Union
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.enities import TrainingParams

SklearnClassificationModel = Union[LogisticRegression, RandomForestClassifier]


def train_model(X_train: np.ndarray, y_train: np.ndarray,
                train_params: TrainingParams
                ) -> SklearnClassificationModel:
    if train_params.model_type == 'LogisticRegression':
        model = LogisticRegression(random_state=train_params.random_state, solver='liblinear')
        param_grid = {'C': np.logspace(-3, 3, 10),
                      'penalty': ['l1', 'l2']}
    elif train_params.model_type == 'RandomForestClassifier':
        model = RandomForestClassifier(random_state=train_params.random_state)
        param_grid = {'max_depth': [3, 5, 7],
                      'n_estimators': [100, 200],
                      'min_samples_leaf': [1, 2, 4],
                      'min_samples_split': [2, 5, 10]
                      }
    else:
        raise NotImplementedError('This model type is not implemented')
    if train_params.grid_search:
        model_grid = GridSearchCV(model, param_grid, scoring='recall', cv=5)
        model_grid.fit(X_train, y_train)
        return model_grid.best_estimator_
    else:
        model.fit(X_train, y_train)
        return model


def predict_model(X: np.ndarray, model: SklearnClassificationModel) -> np.ndarray:
    return model.predict(X)


def evaluate_model(y_pred: np.ndarray, y_target: np.ndarray) -> Dict[str, float]:
    metrics = {'accuracy': accuracy_score(y_pred, y_target),
               'precision': precision_score(y_pred, y_target),
               'recall': recall_score(y_pred, y_target),
               'f1_score': f1_score(y_pred, y_target)}
    return metrics


def save_model(model: object, path_to_model: str):
    with open(path_to_model, 'wb') as f:
        pickle.dump(model, f)
