import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from main import map_yes_or_no

# Test data loading
def test_data_loading():
    df = pd.read_csv('Housing.csv')
    assert not df.empty, "Data loading failed, DataFrame is empty."

# Test mapping of categorical yes/no columns
def test_map_yes_or_no():
    df = pd.DataFrame({
        'mainroad': ['yes', 'no', 'yes'],
        'guestroom': ['no', 'yes', 'no'],
    })
    df_mapped = map_yes_or_no(df, ['mainroad', 'guestroom'])
    assert df_mapped['mainroad'].tolist() == [1, 0, 1], "Mapping of mainroad failed"
    assert df_mapped['guestroom'].tolist() == [0, 1, 0], "Mapping of guestroom failed"

# Test model training
def test_model_training():
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    model = LinearRegression()
    model.fit(X, y)
    assert len(model.coef_) == 5, "Model training failed, wrong number of coefficients"

# Test model evaluation (R2 score)
def test_model_evaluation():
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    assert r2 <= 1.0 and r2 >= 0.0, "Invalid R2 score"

# Test data scaling
def test_data_scaling():
    X = np.random.rand(100, 5)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    assert X_scaled.mean() == pytest.approx(0, abs=1e-6), "Scaling failed, mean is not zero"
    assert np.abs(X_scaled.std() - 1) < 1e-6, "Scaling failed, std is not one"
