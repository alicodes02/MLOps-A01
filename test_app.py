import pytest
from flask import json
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    """Test the home page (index.html) loads successfully."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"House Price Prediction" in response.data  

def test_predict_valid_data(client):
    """Test the /predict endpoint with valid data."""
    data = {
        'area': 1500,
        'bedrooms': 3,
        'bathrooms': 2,
        'stories': 2,
        'mainroad': 'yes',
        'guestroom': 'yes',
        'basement': 'no',
        'hotwaterheating': 'yes',
        'airconditioning': 'yes',
        'parking': 2,
        'prefarea': 'yes',
        'furnishingstatus': 'furnished'
    }
    response = client.post('/predict', json=data)
    assert response.status_code == 200
    response_json = json.loads(response.data)
    assert 'predicted_price' in response_json
    assert isinstance(response_json['predicted_price'], (int, float))

def test_predict_missing_data(client):
    """Test the /predict endpoint with missing data."""
    data = {
        'area': 1500,
        'bedrooms': 3,
        'bathrooms': 2,
        # Missing 'stories'
        'mainroad': 'yes',
        'guestroom': 'yes',
        'basement': 'no',
        'hotwaterheating': 'yes',
        'airconditioning': 'yes',
        'parking': 2,
        'prefarea': 'yes',
        'furnishingstatus': 'furnished'
    }
    response = client.post('/predict', json=data)
    assert response.status_code == 400  
    response_json = json.loads(response.data)
    assert 'error' in response_json
    assert response_json['error'] == 'Missing or invalid data'

def test_predict_invalid_data_type(client):
    """Test the /predict endpoint with invalid data types."""
    data = {
        'area': 'large',  
        'bedrooms': 'three', 
        'bathrooms': 2,
        'stories': 2,
        'mainroad': 'yes',
        'guestroom': 'yes',
        'basement': 'no',
        'hotwaterheating': 'yes',
        'airconditioning': 'yes',
        'parking': 2,
        'prefarea': 'yes',
        'furnishingstatus': 'furnished'
    }
    response = client.post('/predict', json=data)
    assert response.status_code == 400  
    response_json = json.loads(response.data)
    assert 'error' in response_json
    assert response_json['error'] == 'Invalid data types'
