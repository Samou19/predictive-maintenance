
import numpy as np
from src.model import train_model, predict_cycle

def test_train_and_predict():
    X_train = np.random.rand(10, 5)
    y_train = np.random.randint(0, 2, size=10)
    model = train_model(X_train, y_train)
    pred = predict_cycle(model, X_train[0])
    assert pred in [0, 1]
