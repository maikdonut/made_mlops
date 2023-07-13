import os
import pickle
import pandas as pd

from fastapi import FastAPI
from fastapi_health import health
from validator_request import Patient

app = FastAPI()

model = None
transformer = None


@app.on_event('startup')
def load_model():
    path_to_model = os.getenv('PATH_TO_MODEL')
    path_to_transformer = os.getenv('PATH_TO_TRANSFORMER')

    with open(path_to_model, 'rb') as f:
        global model
        model = pickle.load(f)

    with open(path_to_transformer, 'rb') as f:
        global transformer
        transformer = pickle.load(f)


@app.post('/predict')
async def get_predict(data: Patient):
    data_df = pd.DataFrame([data.dict()])
    X = transformer.transform(data_df)
    prediction = model.predict(X)
    condition = 'healthy' if not prediction[0] else 'sick'
    return {'condition': condition}


@app.get('/')
async def root():
    return 'Hello!'


def check_ready():
    return model is not None and transformer is not None


app.add_api_route("/health", health([check_ready]))



