import json
import logging

import pandas as pd
import requests

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
logger.addHandler(handler)

data = pd.read_csv('synthetic_data.csv').drop('condition', axis=1)
data_requests = data.to_dict(orient='records')

for request in data_requests:
    response = requests.post(
        'http://127.0.0.1:8000/predict',
        json.dumps(request)
    )
    logger.info('Response:')
    logger.info(f'Code: {response.status_code}')
    logger.info(f'Message: {response.json()}')