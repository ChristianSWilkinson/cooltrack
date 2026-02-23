import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score
import logging
from .constants import INDEPENDENT_DIMS, PHOTOMETRY_BANDS

class ThermalEvolutionModels:
    def __init__(self):
        self.dsdt_model = None
        self.tint_model = None
        self.radius_model = None
        self.photo_models = {}  # NEW: Dictionary to hold all 15 photometry models

    def train_models(self, df: pd.DataFrame, tune_hyperparameters: bool = False, clean_outliers: bool = True, outlier_threshold: float = 1.0):
        training_df = df.copy()

        if clean_outliers:
            logging.info("Performing first-pass training to identify grid outliers...")
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

        state_features = INDEPENDENT_DIMS + ['S_physical']
        X_state = training_df[state_features]
        
        # --- 1. Train T_int and Radius Models ---
        logging.info("Training T_int and Radius state models...")
        self.tint_model = xgb.XGBRegressor(n_estimators=300, max_depth=7, learning_rate=0.05, n_jobs=-1, random_state=42)
        self.tint_model.fit(X_state, training_df['T_int'])
        
        self.radius_model = xgb.XGBRegressor(n_estimators=300, max_depth=7, learning_rate=0.05, n_jobs=-1, random_state=42)
        self.radius_model.fit(X_state, training_df['Req_Rj'])

        # --- 2. Train Photometry Models (NEW) ---
        logging.info(f"Training {len(PHOTOMETRY_BANDS)} photometric band models (this will take a minute)...")
        for band in PHOTOMETRY_BANDS:
            target_col = f'log_{band}'
            model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, n_jobs=-1, random_state=42)
            model.fit(X_state, training_df[target_col])
            self.photo_models[band] = model
            
        logging.info("All photometry models trained successfully!")

        # --- 3. Train Final dS/dt Model ---
        dsdt_features = INDEPENDENT_DIMS + ['S_physical', 'T_int']
        X_dsdt = training_df[dsdt_features]
        y_dsdt = training_df['abs_log_dsdt']
        
        X_train, X_test, y_train, y_test = train_test_split(X_dsdt, y_dsdt, test_size=0.2, random_state=42)
        
        logging.info("Training final baseline dS/dt model...")
        self.dsdt_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=7, n_jobs=-1, random_state=42)
        self.dsdt_model.fit(X_train, y_train)
            
        preds = self.dsdt_model.predict(X_test)
        logging.info(f"Final dS/dt test R^2: {r2_score(y_test, preds):.4f}")

        return training_df
    
    def save_models(self, save_dir: str):
        """Saves all trained models to the specified directory."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        logging.info(f"Saving models to {save_dir}...")
        self.tint_model.save_model(os.path.join(save_dir, 'tint_model.json'))
        self.radius_model.save_model(os.path.join(save_dir, 'radius_model.json'))
        self.dsdt_model.save_model(os.path.join(save_dir, 'dsdt_model.json'))
        
        for band, model in self.photo_models.items():
            model.save_model(os.path.join(save_dir, f'photo_{band}.json'))
            
        logging.info("All models saved successfully.")

    def load_models(self, load_dir: str):
        """Loads all trained models from the specified directory."""
        import os
        from .constants import PHOTOMETRY_BANDS
        
        logging.info(f"Loading models from {load_dir}...")
        
        self.tint_model = xgb.XGBRegressor()
        self.tint_model.load_model(os.path.join(load_dir, 'tint_model.json'))
        
        self.radius_model = xgb.XGBRegressor()
        self.radius_model.load_model(os.path.join(load_dir, 'radius_model.json'))
        
        self.dsdt_model = xgb.XGBRegressor()
        self.dsdt_model.load_model(os.path.join(load_dir, 'dsdt_model.json'))
        
        self.photo_models = {}
        for band in PHOTOMETRY_BANDS:
            model = xgb.XGBRegressor()
            model.load_model(os.path.join(load_dir, f'photo_{band}.json'))
            self.photo_models[band] = model
            
        logging.info("All models loaded successfully.")