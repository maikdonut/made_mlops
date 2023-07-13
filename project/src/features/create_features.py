import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.enities import FeatureParams


def create_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline([
        (
            "impute",
            SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        ),
        (
            'ohe',
            OneHotEncoder()
        )
    ])
    return categorical_pipeline


def create_numerical_pipeline() -> Pipeline:
    numerical_pipeline = Pipeline([('impute', SimpleImputer(strategy='mean')),
                                   ('scaler', StandardScaler())])
    return numerical_pipeline


def process_features(transformer: ColumnTransformer, data: pd.DataFrame) -> np.ndarray:
    return transformer.transform(data)


def create_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                'categorical_pipeline',
                create_categorical_pipeline(),
                params.categorical_features,
            ),
            (
                'numerical_pipeline',
                create_numerical_pipeline(),
                params.numerical_features,
            ),
        ]
    )
    return transformer
