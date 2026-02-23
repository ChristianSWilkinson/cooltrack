"""
Signal processing and data smoothing module for CoolTrack.

Because tree-based machine learning models (like XGBoost) make predictions using 
discrete step functions, their raw outputs can contain non-physical "staircase" 
artifacts. This module provides a suite of mathematical filters to smooth these 
predictions back into continuous, physically realistic planetary cooling tracks.
"""

import logging

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

# Configure module-level logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class TrackSmoother:
    """
    A collection of smoothing algorithms optimized for planetary cooling tracks.
    """

    @staticmethod
    def savitzky_golay(
        y: np.ndarray, window_length: int = 21, polyorder: int = 3
    ) -> np.ndarray:
        """
        Applies a Savitzky-Golay filter to a 1D array.

        This method is generally best for cooling tracks because it preserves 
        physical shapes and local extrema without aggressively flattening peaks.

        Args:
            y (np.ndarray): The raw, noisy input data.
            window_length (int, optional): The length of the filter window 
                (must be an odd integer). Defaults to 21.
            polyorder (int, optional): The order of the polynomial used to fit 
                the samples. Defaults to 3.

        Returns:
            np.ndarray: The smoothed data array.
        """
        # Ensure the window length is valid and odd
        if len(y) < window_length:
            window_length = max(
                3, len(y) - 1 if len(y) % 2 == 0 else len(y) - 2
            )
            
        return savgol_filter(
            y, window_length=window_length, polyorder=polyorder
        )

    @staticmethod
    def spline(
        x: np.ndarray, 
        y: np.ndarray, 
        smoothing_factor: float = None, 
        degree: int = 3
    ) -> np.ndarray:
        """
        Fits a smooth B-spline to the data.

        Splines are excellent for generating mathematically guaranteed smooth 
        derivatives. This method automatically sorts the input arrays since 
        spline fitting requires strictly increasing independent variables.

        Args:
            x (np.ndarray): The independent variable (e.g., ages).
            y (np.ndarray): The dependent variable (e.g., temperatures).
            smoothing_factor (float, optional): Positive smoothing factor used 
                to choose the number of knots. Defaults to None.
            degree (int, optional): Degree of the smoothing spline. Defaults to 3.

        Returns:
            np.ndarray: The smoothed data array evaluated at x.
        """
        idx = np.argsort(x)
        spline_fit = UnivariateSpline(
            x[idx], y[idx], s=smoothing_factor, k=degree
        )
        return spline_fit(x)

    @staticmethod
    def gaussian(y: np.ndarray, sigma: float = 2.0) -> np.ndarray:
        """
        Applies a 1D Gaussian filter to the data.

        Very aggressive against high-frequency ML noise, but be careful: large 
        sigma values can artificially widen or clip sharp physical transitions.

        Args:
            y (np.ndarray): The raw input data.
            sigma (float, optional): Standard deviation for Gaussian kernel. 
                Defaults to 2.0.

        Returns:
            np.ndarray: The smoothed data array.
        """
        return gaussian_filter1d(y, sigma=sigma)

    @staticmethod
    def moving_average(y: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Calculates a simple rolling moving average.

        The window is center-aligned to prevent artificially phase-shifting 
        the planetary ages forward or backward in time.

        Args:
            y (np.ndarray): The raw input data.
            window_size (int, optional): The number of periods to average over. 
                Defaults to 5.

        Returns:
            np.ndarray: The smoothed data array.
        """
        return pd.Series(y).rolling(
            window=window_size, min_periods=1, center=True
        ).mean().values

    @classmethod
    def smooth(
        cls, x: np.ndarray, y: np.ndarray, method: str = 'savgol', **kwargs
    ) -> np.ndarray:
        """
        Master wrapper to easily swap out smoothing methods on the fly.

        Supported methods are: 'savgol', 'spline', 'gaussian', and 
        'moving_average'. Any additional kwargs are passed directly to the 
        chosen underlying smoothing algorithm.

        Args:
            x (np.ndarray): The independent variable (required for splines).
            y (np.ndarray): The dependent variable to be smoothed.
            method (str, optional): The smoothing algorithm to use. Defaults 
                to 'savgol'.
            **kwargs: Additional keyword arguments for the chosen method.

        Returns:
            np.ndarray: The resulting smoothed array. If an unknown method is 
            provided, returns the raw `y` array and logs a warning.
        """
        if method == 'savgol':
            return cls.savitzky_golay(y, **kwargs)
        elif method == 'spline':
            return cls.spline(x, y, **kwargs)
        elif method == 'gaussian':
            return cls.gaussian(y, **kwargs)
        elif method == 'moving_average':
            return cls.moving_average(y, **kwargs)
        else:
            logging.warning(
                f"Unknown smoothing method '{method}'. Returning raw data."
            )
            return np.array(y)