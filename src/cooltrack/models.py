import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score
import logging
from .constants import INDEPENDENT_DIMS

class ThermalEvolutionModels:
    def __init__(self):
        self.dsdt_model = None
        self.tint_model = None

    def train_models(self, df: pd.DataFrame, tune_hyperparameters: bool = False, clean_outliers: bool = True, outlier_threshold: float = 1.0):
        """
        Trains the T_int and dS/dt models. 
        If clean_outliers is True, it performs a first-pass training to identify 
        and remove numerical noise from the grid before training the final models.
        """
        training_df = df.copy()

        if clean_outliers:
            logging.info("Performing first-pass training to identify grid outliers...")
            # Quick baseline model just to find the broken physics points
            dsdt_features = INDEPENDENT_DIMS + ['S_physical', 'T_int']
            X_temp = training_df[dsdt_features]
            y_temp = training_df['abs_log_dsdt']
            
            temp_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42)
            temp_model.fit(X_temp, y_temp)
            
            all_preds = temp_model.predict(X_temp)
            errors_dex = np.abs(y_temp - all_preds)
            
            bad_mask = errors_dex > outlier_threshold
            outlier_count = bad_mask.sum()
            
            if outlier_count > 0:
                logging.info(f"Dropping {outlier_count} corrupted grid points (> {outlier_threshold} dex error).")
                training_df = training_df[~bad_mask].reset_index(drop=True)
            else:
                logging.info("No massive outliers found in the grid.")

        # --- 1. Train Final T_int Model ---
        tint_features = INDEPENDENT_DIMS + ['S_physical']
        X_tint = training_df[tint_features]
        y_tint = training_df['T_int']
        
        logging.info("Training final T_int state model...")
        self.tint_model = xgb.XGBRegressor(n_estimators=300, max_depth=7, learning_rate=0.05, n_jobs=-1, random_state=42)
        self.tint_model.fit(X_tint, y_tint)
        logging.info(f"Final T_int R^2: {self.tint_model.score(X_tint, y_tint):.4f}")

        # --- 2. Train Final dS/dt Model ---
        dsdt_features = INDEPENDENT_DIMS + ['S_physical', 'T_int']
        X_dsdt = training_df[dsdt_features]
        y_dsdt = training_df['abs_log_dsdt']
        
        X_train, X_test, y_train, y_test = train_test_split(X_dsdt, y_dsdt, test_size=0.2, random_state=42)
        
        if tune_hyperparameters:
            logging.info("Tuning dS/dt model hyperparameters...")
            # Basic grid for tuning
            param_dist = {
                'n_estimators': [300, 500],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.01, 0.05]
            }
            search = RandomizedSearchCV(
                xgb.XGBRegressor(random_state=42, n_jobs=-1),
                param_distributions=param_dist, n_iter=5, cv=3, scoring='r2', n_jobs=-1, verbose=1
            )
            search.fit(X_train, y_train)
            self.dsdt_model = search.best_estimator_
            logging.info(f"Best params: {search.best_params_}")
        else:
            logging.info("Training final baseline dS/dt model...")
            self.dsdt_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=7, n_jobs=-1, random_state=42)
            self.dsdt_model.fit(X_train, y_train)
            
        preds = self.dsdt_model.predict(X_test)
        logging.info(f"Final dS/dt test R^2: {r2_score(y_test, preds):.4f}")

        return training_df