import json

import pytest
from fastapi.testclient import TestClient
from main import app, load_model

client = TestClient(app)


@pytest.fixture(scope='session', autouse=True)
def initialize_model():
    load_model()


def test_good_request():
    request = {
        'age': 69,
        'sex': 1,
        'cp': 0,
        'trestbps': 160,
        'chol': 234,
        'fbs': 1,
        'restecg': 2,
        'thalach': 131,
        'exang': 0,
        'oldpeak': 0.1,
        'slope': 1,
        'ca': 1,
        'thal': 0,
    }
    response = client.post(
        url='/predict',
        content=json.dumps(request)
    )
    assert response.status_code == 200
    assert response.json() == {'condition': 'healthy'}


def test_health_model():
    response = client.get('/health')
    assert response.status_code == 200


def test_missed_field_data():
    request = {
        'sex': 1,
        'cp': 0,
        'trestbps': 160,
        'chol': 234,
        'fbs': 1,
        'restecg': 2,
        'thalach': 131,
        'exang': 0,
        'oldpeak': 0.1,
        'slope': 1,
        'ca': 1,
        'thal': 0,
    }
    response = client.post(
        url='/predict',
        content=json.dumps(request)
    )
    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == 'field required'


def test_wrong_numerical_value():
    request = {
        'age': 500,
        'sex': 1,
        'cp': 0,
        'trestbps': 160,
        'chol': 234,
        'fbs': 1,
        'restecg': 2,
        'thalach': 131,
        'exang': 0,
        'oldpeak': 0.1,
        'slope': 1,
        'ca': 1,
        'thal': 0,
    }
    response = client.post(
        url='/predict',
        content=json.dumps(request)
    )
    assert response.status_code == 400


def test_wrong_literal_value():
    request = {
        'age': 69,
        'sex': 5,
        'cp': 0,
        'trestbps': 160,
        'chol': 234,
        'fbs': 1,
        'restecg': 2,
        'thalach': 131,
        'exang': 0,
        'oldpeak': 0.1,
        'slope': 1,
        'ca': 1,
        'thal': 0,
    }
    response = client.post(
        url='/predict',
        content=json.dumps(request)
    )
    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == 'unexpected value; permitted: 0, 1'
