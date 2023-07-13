from dataclasses import dataclass
import yaml
from marshmallow_dataclass import class_schema

from .feature_params import FeatureParams


@dataclass()
class PredictPipelineParams:
    path_to_data: str
    path_to_predictions: str
    path_to_model: str
    feature_params: FeatureParams


PredictPipelineParamsSchema = class_schema(PredictPipelineParams)


def read_predict_pipeline_params(config_path: str) -> PredictPipelineParams:
    with open(config_path, "r") as config:
        schema = PredictPipelineParamsSchema()
        return schema.load(yaml.safe_load(config))
