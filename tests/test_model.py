
import pytest
#from src.model import get_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_model(model_type="logistic"):
    """
    Retourne un modèle en fonction du type choisi.
    model_type : 'logistic', 'random_forest', 'xgboost'
    """
    if model_type == "logistic":
        return LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "xgboost":
        return XGBClassifier(eval_metric='logloss', random_state=42)
    else:
        raise ValueError(f"Type de modèle inconnu : {model_type}")
    

def test_get_model_logistic():
    model = get_model("logistic")
    assert isinstance(model, LogisticRegression)

def test_get_model_random_forest():
    model = get_model("random_forest")
    assert isinstance(model, RandomForestClassifier)

def test_get_model_xgboost():
    model = get_model("xgboost")
    assert isinstance(model, XGBClassifier)

def test_get_model_invalid_type():
    with pytest.raises(ValueError):
        get_model("invalid_type")
