import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score
from scipy.stats import randint, uniform
import logging
from .constants import INDEPENDENT_DIMS

class ThermalEvolutionModels:
    def __init__(self):
        self.dsdt_model = None
        self.tint_model = None

    def train_models(self, df, tune_hyperparameters=False):
        """Trains both the T_int state model and the dS/dt derivative model."""
        
        # 1. Train T_int model (Predicts T_int given state + Entropy)
        tint_features = INDEPENDENT_DIMS + ['S_physical']
        X_tint = df[tint_features]
        y_tint = df['T_int']
        
        logging.info("Training T_int state model...")
        self.tint_model = xgb.XGBRegressor(n_estimators=300, max_depth=7, learning_rate=0.05, n_jobs=-1, random_state=42)
        self.tint_model.fit(X_tint, y_tint)
        logging.info(f"T_int model R^2: {self.tint_model.score(X_tint, y_tint):.4f}")

        # 2. Train dS/dt model (Predicts cooling rate given full state)
        dsdt_features = INDEPENDENT_DIMS + ['S_physical', 'T_int']
        X_dsdt = df[dsdt_features]
        y_dsdt = df['abs_log_dsdt']
        
        X_train, X_test, y_train, y_test = train_test_split(X_dsdt, y_dsdt, test_size=0.2, random_state=42)
        
        if tune_hyperparameters:
            logging.info("Tuning dS/dt model hyperparameters...")
            param_dist = {
                'n_estimators': randint(200, 800),
                'max_depth': randint(5, 12),
                'learning_rate': uniform(0.01, 0.2),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4)
            }
            search = RandomizedSearchCV(
                xgb.XGBRegressor(random_state=42, n_jobs=-1),
                param_distributions=param_dist, n_iter=20, cv=3, scoring='r2', n_jobs=-1, verbose=1
            )
            search.fit(X_train, y_train)
            self.dsdt_model = search.best_estimator_
            logging.info(f"Best params: {search.best_params_}")
        else:
            logging.info("Training baseline dS/dt model...")
            self.dsdt_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=7, n_jobs=-1, random_state=42)
            self.dsdt_model.fit(X_train, y_train)
            
        preds = self.dsdt_model.predict(X_test)
        logging.info(f"dS/dt model test R^2: {r2_score(y_test, preds):.4f}")