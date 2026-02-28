"""Stability checks and walk-forward evaluation.

Provides utilities for:
1. Walk-forward train/test splits
2. Regime-based evaluation (by volatility, etc.)
3. Performance stability metrics
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Iterator, Callable, Any
import pandas as pd
import numpy as np


def walk_forward_split(
    data: pd.DataFrame,
    n_splits: int = 5,
    min_train_size: timedelta = timedelta(hours=1),
    test_size: timedelta = timedelta(minutes=30),
    gap: timedelta = timedelta(minutes=5),
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate walk-forward train/test splits.
    
    Each split uses all data up to point t for training,
    and data from t+gap to t+gap+test_size for testing.
    
    Args:
        data: DataFrame with datetime index.
        n_splits: Number of splits to generate.
        min_train_size: Minimum training data duration.
        test_size: Duration of test set.
        gap: Gap between train and test (prevents leakage).
    
    Yields:
        Tuples of (train_data, test_data).
    """
    if len(data) < 2:
        return
    
    start_time = data.index.min()
    end_time = data.index.max()
    total_duration = end_time - start_time
    
    # Calculate step size
    available_test_time = total_duration - min_train_size - gap - test_size
    if available_test_time.total_seconds() <= 0:
        raise ValueError("Not enough data for requested split parameters")
    
    step = available_test_time / n_splits
    
    for i in range(n_splits):
        split_point = start_time + min_train_size + i * step
        
        train_end = split_point
        test_start = split_point + gap
        test_end = test_start + test_size
        
        if test_end > end_time:
            break
        
        train_mask = data.index <= train_end
        test_mask = (data.index >= test_start) & (data.index <= test_end)
        
        train_data = data[train_mask]
        test_data = data[test_mask]
        
        if len(train_data) > 0 and len(test_data) > 0:
            yield train_data, test_data


def time_series_split(
    data: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    gap: timedelta = timedelta(minutes=1),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simple time-based train/val/test split.
    
    Args:
        data: DataFrame with datetime index.
        train_frac: Fraction for training.
        val_frac: Fraction for validation.
        gap: Gap between splits.
    
    Returns:
        Tuple of (train, val, test) DataFrames.
    """
    start_time = data.index.min()
    end_time = data.index.max()
    total_duration = end_time - start_time
    
    train_end = start_time + total_duration * train_frac
    val_start = train_end + gap
    val_end = val_start + total_duration * val_frac
    test_start = val_end + gap
    
    train = data[data.index <= train_end]
    val = data[(data.index >= val_start) & (data.index <= val_end)]
    test = data[data.index >= test_start]
    
    return train, val, test


def regime_split_by_volatility(
    data: pd.DataFrame,
    vol_col: str = "realized_vol_20",
    n_regimes: int = 3,
    labels: List[str] = None,
) -> pd.Series:
    """Split data into regimes based on volatility quantiles.
    
    Args:
        data: DataFrame with features.
        vol_col: Column name for volatility measure.
        n_regimes: Number of regimes (2 or 3 recommended).
        labels: Labels for regimes (low, medium, high for n=3).
    
    Returns:
        Series with regime labels.
    """
    if labels is None:
        if n_regimes == 2:
            labels = ["low_vol", "high_vol"]
        elif n_regimes == 3:
            labels = ["low_vol", "medium_vol", "high_vol"]
        else:
            labels = [f"regime_{i}" for i in range(n_regimes)]
    
    vol = data[vol_col]
    
    # Use quantiles to define regimes
    quantiles = np.linspace(0, 1, n_regimes + 1)
    thresholds = [vol.quantile(q) for q in quantiles[1:-1]]
    
    regimes = pd.Series(index=data.index, dtype=str)
    
    regimes[vol <= thresholds[0]] = labels[0]
    
    for i in range(1, len(thresholds)):
        mask = (vol > thresholds[i - 1]) & (vol <= thresholds[i])
        regimes[mask] = labels[i]
    
    regimes[vol > thresholds[-1]] = labels[-1]
    
    return regimes


def regime_split_by_trend(
    data: pd.DataFrame,
    price_col: str = "mid_price",
    window: int = 20,
    labels: List[str] = None,
) -> pd.Series:
    """Split data into regimes based on price trend.
    
    Args:
        data: DataFrame with features.
        price_col: Column name for price.
        window: Window for trend calculation.
        labels: Labels for regimes.
    
    Returns:
        Series with regime labels.
    """
    if labels is None:
        labels = ["downtrend", "sideways", "uptrend"]
    
    # Compute trend using momentum
    momentum = data[price_col].pct_change(window)
    
    # Define thresholds
    up_threshold = momentum.quantile(0.67)
    down_threshold = momentum.quantile(0.33)
    
    regimes = pd.Series(index=data.index, dtype=str)
    regimes[momentum <= down_threshold] = labels[0]
    regimes[(momentum > down_threshold) & (momentum < up_threshold)] = labels[1]
    regimes[momentum >= up_threshold] = labels[2]
    
    return regimes


def evaluate_by_regime(
    data: pd.DataFrame,
    regime_labels: pd.Series,
    metric_fn: Callable[[pd.DataFrame], Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Evaluate metrics separately for each regime.
    
    Args:
        data: DataFrame with predictions/performance data.
        regime_labels: Series with regime labels.
        metric_fn: Function that computes metrics from data subset.
    
    Returns:
        Dictionary mapping regime to metrics.
    """
    results = {}
    
    for regime in regime_labels.unique():
        if pd.isna(regime):
            continue
        
        mask = regime_labels == regime
        regime_data = data[mask]
        
        if len(regime_data) > 0:
            results[str(regime)] = metric_fn(regime_data)
    
    return results


def compute_stability_metrics(
    walk_forward_results: List[Dict[str, float]],
    metric_name: str = "sharpe_ratio",
) -> Dict[str, float]:
    """Compute stability metrics across walk-forward runs.
    
    Args:
        walk_forward_results: List of metric dictionaries from each split.
        metric_name: Primary metric to check stability.
    
    Returns:
        Stability metrics.
    """
    if not walk_forward_results:
        return {}
    
    values = [r.get(metric_name, np.nan) for r in walk_forward_results]
    values = [v for v in values if not np.isnan(v)]
    
    if not values:
        return {}
    
    return {
        f"{metric_name}_mean": np.mean(values),
        f"{metric_name}_std": np.std(values),
        f"{metric_name}_min": np.min(values),
        f"{metric_name}_max": np.max(values),
        f"{metric_name}_range": np.max(values) - np.min(values),
        f"{metric_name}_cv": np.std(values) / abs(np.mean(values)) if np.mean(values) != 0 else np.inf,
    }


def detect_regime_shift(
    data: pd.DataFrame,
    feature_cols: List[str],
    window: int = 100,
    threshold: float = 2.0,
) -> pd.Series:
    """Detect potential regime shifts using z-scores.
    
    Args:
        data: DataFrame with features.
        feature_cols: Features to monitor.
        window: Rolling window for baseline.
        threshold: Z-score threshold for flagging shift.
    
    Returns:
        Boolean series indicating regime shifts.
    """
    shifts = pd.Series(False, index=data.index)
    
    for col in feature_cols:
        if col not in data.columns:
            continue
        
        # Compute rolling statistics
        roll_mean = data[col].rolling(window=window).mean()
        roll_std = data[col].rolling(window=window).std()
        
        # Compute z-score of current value vs rolling window
        zscore = (data[col] - roll_mean) / roll_std
        
        # Flag shifts
        shifts |= abs(zscore) > threshold
    
    return shifts


def cross_validation_score_time_series(
    data: pd.DataFrame,
    model_fn: Callable[[pd.DataFrame, pd.DataFrame], Any],
    score_fn: Callable[[Any, pd.DataFrame], float],
    n_splits: int = 5,
) -> List[float]:
    """Time-series cross-validation scoring.
    
    Args:
        data: Full dataset.
        model_fn: Function that trains model on train data and returns model.
        score_fn: Function that scores model on test data.
        n_splits: Number of CV splits.
    
    Returns:
        List of scores for each split.
    """
    scores = []
    
    for train_data, test_data in walk_forward_split(data, n_splits=n_splits):
        model = model_fn(train_data, test_data)
        score = score_fn(model, test_data)
        scores.append(score)
    
    return scores
