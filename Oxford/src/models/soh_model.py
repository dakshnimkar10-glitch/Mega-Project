import os
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

class SOHEstimator:
    """
    State of Health (SOH %) Machine Learning Estimator.
    Trained on cycle-level degradation features (charging duration, temp rise, cycle count).
    """
    def __init__(self, n_estimators=100, max_depth=5, random_state=42):
        try:
            self.model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.03,
                random_state=random_state,
                n_jobs=-1
            )
            self.model_type = 'XGBoost'
        except Exception:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
            self.model_type = 'RandomForest'
            
        self.feature_names = None

    def fit(self, X, y):
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        return self

    def predict(self, X):
        preds = self.model.predict(X)
        return np.clip(preds, 0.0, 100.0)

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'predictions': preds
        }

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({'model': self.model, 'feature_names': self.feature_names, 'type': self.model_type}, filepath)

    @classmethod
    def load(cls, filepath):
        data = joblib.load(filepath)
        obj = cls()
        obj.model = data['model']
        obj.feature_names = data['feature_names']
        obj.model_type = data['type']
        return obj
