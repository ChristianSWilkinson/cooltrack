import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TrackSmoother:
    """
    A collection of smoothing algorithms optimized for planetary cooling tracks.
    """
    @staticmethod
    def savitzky_golay(y: np.ndarray, window_length: int = 21, polyorder: int = 3) -> np.ndarray:
        """
        Best for preserving physical shapes and local extrema without flattening peaks.
        window_length must be an odd number.
        """
        if len(y) < window_length:
            window_length = max(3, len(y) - 1 if len(y) % 2 == 0 else len(y) - 2)
        return savgol_filter(y, window_length=window_length, polyorder=polyorder)

    @staticmethod
    def spline(x: np.ndarray, y: np.ndarray, smoothing_factor: float = None, degree: int = 3) -> np.ndarray:
        """
        Fits a smooth B-spline. Excellent for generating guaranteed smooth derivatives.
        Requires strictly increasing x values, so we sort internally.
        """
        idx = np.argsort(x)
        spline_fit = UnivariateSpline(x[idx], y[idx], s=smoothing_factor, k=degree)
        return spline_fit(x)

    @staticmethod
    def gaussian(y: np.ndarray, sigma: float = 2.0) -> np.ndarray:
        """
        Applies a 1D Gaussian filter. Very aggressive against high-frequency ML noise, 
        but can artificially widen or clip sharp physical transitions.
        """
        return gaussian_filter1d(y, sigma=sigma)

    @staticmethod
    def moving_average(y: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        A simple rolling average. Centered to prevent phase shifting the ages.
        """
        return pd.Series(y).rolling(window=window_size, min_periods=1, center=True).mean().values

    @classmethod
    def smooth(cls, x: np.ndarray, y: np.ndarray, method: str = 'savgol', **kwargs) -> np.ndarray:
        """
        Master wrapper to easily swap out smoothing methods on the fly.
        Supported methods: 'savgol', 'spline', 'gaussian', 'moving_average'
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
            logging.warning(f"Unknown smoothing method '{method}'. Returning raw data.")
            return np.array(y)