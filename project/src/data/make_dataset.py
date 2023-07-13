from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.enities import SplittingParams


def read_data(data_path: str) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    return data


def extract_target(data: pd.DataFrame,
                   target_col_name: str) -> Tuple[pd.DataFrame, np.ndarray]:
    X = data.drop(target_col_name, axis=1)
    y = data[target_col_name].to_numpy()
    return X, y


def split_train_test_data(X: pd.DataFrame, y: np.ndarray, params: SplittingParams
                          ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=params.val_size, random_state=params.random_state
    )
    return X_train, X_val, y_train, y_val
