
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_model(X_train, y_train, model_path="model.pkl"):
    #model = RandomForestClassifier(n_estimators=100, random_state=42)
    model = XGBClassifier(eval_metric='logloss')
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    return model

def load_model(model_path="model.pkl"):
    return joblib.load(model_path)

def predict_cycle(model, X_cycle):
    return model.predict([X_cycle])[0]

