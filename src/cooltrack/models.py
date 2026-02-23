"""
Machine learning surrogate models for CoolTrack.

This module provides the ThermalEvolutionModels class, which acts as an 
ensemble of XGBoost regressors. It learns to predict a planet's cooling 
rate (dS/dt), internal temperature, radius, and photometric fluxes directly 
from the physical grid, effectively replacing the numerical 1D structural 
evolution codes with a fast, differentiable surrogate.
"""

import logging
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from .constants import INDEPENDENT_DIMS, PHOTOMETRY_BANDS

# Configure module-level logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class ThermalEvolutionModels:
    """
    Manages the training, saving, and loading of the XGBoost surrogate models.

    This class handles a multi-output prediction pipeline. It first identifies 
    and filters out numerical artifacts from the training grid, then trains 
    dedicated models for internal temperature, planetary radius, cooling rate 
    (dS/dt), and JWST photometric fluxes.

    Attributes:
        dsdt_model (xgb.XGBRegressor): Predicts the log absolute cooling rate.
        tint_model (xgb.XGBRegressor): Predicts the internal temperature (K).
        radius_model (xgb.XGBRegressor): Predicts the planetary radius (R_J).
        photo_models (dict): A dictionary mapping photometric band names (str) 
            to their trained XGBRegressor models.
    """

    def __init__(self):
        """Initializes the ML ensemble with empty model placeholders."""
        self.dsdt_model = None
        self.tint_model = None
        self.radius_model = None
        self.photo_models = {}

    def train_models(
        self, 
        df: pd.DataFrame, 
        tune_hyperparameters: bool = False, 
        clean_outliers: bool = True, 
        outlier_threshold: float = 1.0
    ) -> pd.DataFrame:
        """
        Trains the full suite of surrogate models on the provided grid data.

        Optionally performs a first-pass training run to identify and discard 
        grid points that severely violate local thermodynamic trends 
        (usually artifacts of the 1D structural solver failing to converge).

        Args:
            df (pd.DataFrame): The preprocessed training data.
            tune_hyperparameters (bool, optional): Reserved for future 
                hyperparameter tuning functionality. Defaults to False.
            clean_outliers (bool, optional): Whether to drop points with high 
                residuals during a preliminary training pass. Defaults to True.
            outlier_threshold (float, optional): The residual threshold in dex 
                above which a grid point is considered broken. Defaults to 1.0.

        Returns:
            pd.DataFrame: The cleaned DataFrame used for final model training.
        """
        training_df = df.copy()

        if clean_outliers:
            logging.info(
                "Performing first-pass training to identify grid outliers..."
            )
            dsdt_features = INDEPENDENT_DIMS + ['S_physical', 'T_int']
            X_temp = training_df[dsdt_features]
            y_temp = training_df['abs_log_dsdt']
            
            temp_model = xgb.XGBRegressor(
                n_estimators=100, 
                max_depth=6, 
                n_jobs=-1, 
                random_state=42
            )
            temp_model.fit(X_temp, y_temp)
            
            all_preds = temp_model.predict(X_temp)
            errors_dex = np.abs(y_temp - all_preds)
            
            bad_mask = errors_dex > outlier_threshold
            outlier_count = bad_mask.sum()
            
            if outlier_count > 0:
                logging.info(
                    f"Dropping {outlier_count} corrupted grid points "
                    f"(> {outlier_threshold} dex error)."
                )
                training_df = training_df[~bad_mask].reset_index(drop=True)
            else:
                logging.info("No massive outliers found in the grid.")

        state_features = INDEPENDENT_DIMS + ['S_physical']
        X_state = training_df[state_features]
        
        # --- 1. Train T_int and Radius Models ---
        logging.info("Training T_int and Radius state models...")
        self.tint_model = xgb.XGBRegressor(
            n_estimators=300, 
            max_depth=7, 
            learning_rate=0.05, 
            n_jobs=-1, 
            random_state=42
        )
        self.tint_model.fit(X_state, training_df['T_int'])
        
        self.radius_model = xgb.XGBRegressor(
            n_estimators=300, 
            max_depth=7, 
            learning_rate=0.05, 
            n_jobs=-1, 
            random_state=42
        )
        self.radius_model.fit(X_state, training_df['Req_Rj'])

        # --- 2. Train Photometry Models ---
        logging.info(
            f"Training {len(PHOTOMETRY_BANDS)} photometric band models "
            "(this will take a minute)..."
        )
        for band in PHOTOMETRY_BANDS:
            target_col = f'log_{band}'
            model = xgb.XGBRegressor(
                n_estimators=200, 
                max_depth=6, 
                learning_rate=0.05, 
                n_jobs=-1, 
                random_state=42
            )
            model.fit(X_state, training_df[target_col])
            self.photo_models[band] = model
            
        logging.info("All photometry models trained successfully!")

        # --- 3. Train Final dS/dt Model ---
        dsdt_features = INDEPENDENT_DIMS + ['S_physical', 'T_int']
        X_dsdt = training_df[dsdt_features]
        y_dsdt = training_df['abs_log_dsdt']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_dsdt, y_dsdt, test_size=0.2, random_state=42
        )
        
        logging.info("Training final baseline dS/dt model...")
        self.dsdt_model = xgb.XGBRegressor(
            n_estimators=500, 
            learning_rate=0.05, 
            max_depth=7, 
            n_jobs=-1, 
            random_state=42
        )
        self.dsdt_model.fit(X_train, y_train)
            
        preds = self.dsdt_model.predict(X_test)
        logging.info(f"Final dS/dt test R^2: {r2_score(y_test, preds):.4f}")

        return training_df
    
    def save_models(self, save_dir: str):
        """
        Saves all trained XGBoost models to the specified directory.

        Args:
            save_dir (str): The local directory path where the JSON models 
                will be saved. Created if it does not exist.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        logging.info(f"Saving models to {save_dir}...")
        self.tint_model.save_model(os.path.join(save_dir, 'tint_model.json'))
        
        self.radius_model.save_model(
            os.path.join(save_dir, 'radius_model.json')
        )
        
        self.dsdt_model.save_model(os.path.join(save_dir, 'dsdt_model.json'))
        
        for band, model in self.photo_models.items():
            model.save_model(os.path.join(save_dir, f'photo_{band}.json'))
            
        logging.info("All models saved successfully.")

    def load_models(self, load_dir: str):
        """
        Loads all trained XGBoost models from the specified directory.

        Args:
            load_dir (str): The local directory path containing the JSON models.
        """
        logging.info(f"Loading models from {load_dir}...")
        
        self.tint_model = xgb.XGBRegressor()
        self.tint_model.load_model(os.path.join(load_dir, 'tint_model.json'))
        
        self.radius_model = xgb.XGBRegressor()
        self.radius_model.load_model(
            os.path.join(load_dir, 'radius_model.json')
        )
        
        self.dsdt_model = xgb.XGBRegressor()
        self.dsdt_model.load_model(os.path.join(load_dir, 'dsdt_model.json'))
        
        self.photo_models = {}
        for band in PHOTOMETRY_BANDS:
            model = xgb.XGBRegressor()
            model.load_model(os.path.join(load_dir, f'photo_{band}.json'))
            self.photo_models[band] = model
            
        logging.info("All models loaded successfully.")