import numpy as np
from model_pipeline import load_data, train_model

def test_robustness():
    X_train, X_test, y_train, y_test, _ = load_data()
    model = train_model(X_train, y_train)

    sample = X_test[0]
    noisy = sample + np.random.normal(0, 0.1, size=sample.shape)

    pred = model.predict([noisy])[0]
    assert pred in [0, 1]
