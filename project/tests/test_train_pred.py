import os
from generate_synthetic_data import generate_synthetic_data
from src.enities import TrainConfig, SplittingParams, FeatureParams, TrainingParams, PredictPipelineParams
from train_pipeline import train_pipeline
from predict_pipeline import predict_pipeline

cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
generate_synthetic_data()


def test_train_pipeline():
    path_to_data = os.path.join(
        'tests', 'test_data', 'test_train.csv'
    )
    path_to_output = os.path.join(
        'tests', 'test_results', 'test_model.pkl'
    )
    path_to_metrics = os.path.join(
        'tests', 'test_results', 'test_metrics.json'
    )
    splitting_params = SplittingParams(0.2, 17)
    feature_params = FeatureParams(cat_cols, num_cols, 'condition')
    train_params = TrainingParams('LogisticRegression')

    train_pipeline_params = TrainConfig(
        path_to_data=path_to_data,
        path_to_model=path_to_output,
        path_to_metrics=path_to_metrics,
        splitting_params=splitting_params,
        feature_params=feature_params,
        train_params=train_params
    )
    assert not os.path.exists(path_to_output), f'{path_to_output} exists'
    assert not os.path.exists(path_to_metrics), f'{path_to_metrics} exists'
    metrics = train_pipeline(train_pipeline_params)
    for metric, score in metrics.items():
        assert 0 <= score <= 1, f'{metric} is a bad value'


def test_predict_pipeline():
    path_to_data = os.path.join('tests', 'test_data', 'test_predict.csv')
    path_to_output = os.path.join('tests', 'test_results', 'test_res_predict.csv')
    path_to_model = os.path.join('tests', 'test_results', 'test_model.pkl')

    feature_params = FeatureParams(cat_cols, num_cols, 'condition')

    predict_pipeline_params = PredictPipelineParams(
        path_to_data=path_to_data,
        path_to_predictions=path_to_output,
        path_to_model=path_to_model,
        feature_params=feature_params
    )
    assert not os.path.exists(path_to_output), f'{path_to_output} exists'
    predict_pipeline(predict_pipeline_params)
