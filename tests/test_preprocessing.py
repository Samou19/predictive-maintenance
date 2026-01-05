# Tests pour la fonction extract_features
import pandas as pd
from src.model import extract_features

def test_extract_features():
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
    features = extract_features(df, "test")
    assert "test_mean" in features.columns
    assert "test_std" in features.columns
    assert "test_min" in features.columns
    assert "test_max" in features.columns
    assert features.shape[0] == 2

