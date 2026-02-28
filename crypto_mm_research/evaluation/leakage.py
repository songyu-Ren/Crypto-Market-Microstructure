"""Leakage control utilities for time series.

Leakage (lookahead bias) occurs when information from the future
is used to make decisions at time t. In market microstructure,
this is particularly dangerous because:
1. Features may implicitly include future information
2. Labels (targets) may be known before they should be
3. Train/test splits may overlap in time

This module provides utilities to detect and prevent leakage.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


def validate_no_lookahead(
    features: pd.DataFrame,
    target: pd.Series,
    feature_time_col: str = "timestamp",
    target_lag: timedelta = timedelta(seconds=1),
) -> bool:
    """Validate that features don't use future target information.
    
    This checks that all feature timestamps are strictly before
    their corresponding target timestamps (minus lag).
    
    Args:
        features: Feature DataFrame with timestamp index or column.
        target: Target series with timestamp index.
        feature_time_col: Column name if timestamp is not index.
        target_lag: Minimum lag required between feature and target.
    
    Returns:
        True if no lookahead detected.
    
    Raises:
        ValueError: If lookahead is detected.
    """
    # Get feature timestamps
    if feature_time_col in features.columns:
        feature_times = pd.to_datetime(features[feature_time_col])
    else:
        feature_times = features.index
    
    # Get target timestamps
    target_times = target.index
    
    # Align and check
    aligned_features, aligned_target = features.align(target, join="inner")
    
    if feature_time_col in aligned_features.columns:
        ft = pd.to_datetime(aligned_features[feature_time_col])
    else:
        ft = aligned_features.index
    
    tt = aligned_target.index
    
    # Check that feature time + lag <= target time
    for i, (f_time, t_time) in enumerate(zip(ft, tt)):
        if f_time + target_lag > t_time:
            raise ValueError(
                f"Lookahead detected at index {i}: "
                f"feature time {f_time} + lag {target_lag} > target time {t_time}"
            )
    
    return True


def validate_feature_timestamps(
    events: List[Any],
    features: pd.DataFrame,
    max_delay: timedelta = timedelta(seconds=5),
) -> bool:
    """Validate that features are computed without delay.
    
    Ensures feature timestamps match event timestamps within tolerance.
    
    Args:
        events: List of events used to compute features.
        features: Feature DataFrame.
        max_delay: Maximum allowed delay between event and feature.
    
    Returns:
        True if valid.
    
    Raises:
        ValueError: If timestamps don't match.
    """
    from crypto_mm_research.data.events import L2BookSnapshotEvent
    
    # Get event timestamps
    event_times = [
        e.timestamp for e in events
        if isinstance(e, L2BookSnapshotEvent)
    ]
    
    feature_times = features.index
    
    # Check alignment
    if len(event_times) != len(feature_times):
        # This is OK if we subsample features
        pass
    
    # Check that all feature times exist in events
    for ft in feature_times:
        # Find closest event time
        closest_delay = min(
            abs((ft - et).total_seconds()) for et in event_times
        )
        if closest_delay > max_delay.total_seconds():
            raise ValueError(
                f"Feature timestamp {ft} not within {max_delay} of any event"
            )
    
    return True


def shift_labels(
    labels: pd.Series,
    shift_periods: int = 1,
    fill_value: Any = np.nan,
) -> pd.Series:
    """Shift labels forward to prevent lookahead.
    
    When predicting future returns, the label at time t should be
    the return from t to t+1. When training, we must ensure that
    features at time t don't include the label (which would be known
    at t+1). This function shifts labels forward so they align with
    the features that predicted them.
    
    Args:
        labels: Label series (e.g., future returns).
        shift_periods: Number of periods to shift forward.
        fill_value: Value to fill at the end.
    
    Returns:
        Shifted label series.
    """
    return labels.shift(shift_periods, fill_value=fill_value)


def create_lagged_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int],
) -> pd.DataFrame:
    """Create lagged features (past values).
    
    This is safe because it only uses past information.
    
    Args:
        df: Input DataFrame.
        columns: Columns to lag.
        lags: List of lag periods (positive = past).
    
    Returns:
        DataFrame with lagged features added.
    """
    result = df.copy()
    
    for col in columns:
        for lag in lags:
            if lag > 0:
                result[f"{col}_lag{lag}"] = df[col].shift(lag)
    
    return result


def create_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int],
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """Create rolling statistics features.
    
    Uses only past values (rolling window with closed='left').
    
    Args:
        df: Input DataFrame.
        columns: Columns to compute rolling stats on.
        windows: List of window sizes.
        min_periods: Minimum periods for valid result.
    
    Returns:
        DataFrame with rolling features added.
    """
    result = df.copy()
    
    for col in columns:
        for window in windows:
            # Use shift(1) to ensure we only use past values
            rolled = df[col].shift(1).rolling(
                window=window,
                min_periods=min_periods or window // 2,
            )
            result[f"{col}_roll_mean_{window}"] = rolled.mean()
            result[f"{col}_roll_std_{window}"] = rolled.std()
    
    return result


def check_for_target_leakage(
    features: pd.DataFrame,
    target: pd.Series,
    threshold: float = 0.99,
) -> List[str]:
    """Check if any feature is highly correlated with target.
    
    High correlation may indicate target leakage.
    
    Args:
        features: Feature DataFrame.
        target: Target series.
        threshold: Correlation threshold for flagging.
    
    Returns:
        List of potentially leaky feature names.
    """
    leaky_features = []
    
    for col in features.columns:
        if features[col].dtype in ["float64", "float32", "int64", "int32"]:
            corr = features[col].corr(target)
            if abs(corr) > threshold:
                leaky_features.append((col, corr))
    
    return leaky_features


def assert_no_overlap(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    time_buffer: timedelta = timedelta(minutes=1),
) -> bool:
    """Assert that train and test sets don't overlap in time.
    
    Args:
        train_data: Training data.
        test_data: Test data.
        time_buffer: Minimum gap required.
    
    Returns:
        True if no overlap.
    
    Raises:
        ValueError: If overlap detected.
    """
    train_end = train_data.index.max()
    test_start = test_data.index.min()
    
    if train_end + time_buffer > test_start:
        raise ValueError(
            f"Train/test overlap: train ends at {train_end}, "
            f"test starts at {test_start}, buffer {time_buffer}"
        )
    
    return True
