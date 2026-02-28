"""Tests for evaluation utilities."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from crypto_mm_research.evaluation.leakage import (
    validate_no_lookahead,
    shift_labels,
    create_lagged_features,
    create_rolling_features,
    check_for_target_leakage,
    assert_no_overlap,
)
from crypto_mm_research.evaluation.stability import (
    walk_forward_split,
    time_series_split,
    regime_split_by_volatility,
    regime_split_by_trend,
    compute_stability_metrics,
)


class TestLeakageControl:
    """Tests for leakage control utilities."""
    
    def test_validate_no_lookahead_valid(self):
        """Test validation passes with valid data."""
        timestamps = pd.date_range("2024-01-01", periods=10, freq="1min")
        features = pd.DataFrame(
            {"feat1": range(10)},
            index=timestamps,
        )
        target = pd.Series(range(10), index=timestamps)
        
        # Should pass (same timestamps, no lookahead)
        assert validate_no_lookahead(features, target)
    
    def test_shift_labels(self):
        """Test label shifting."""
        labels = pd.Series([1, 2, 3, 4, 5])
        shifted = shift_labels(labels, shift_periods=1)
        
        # First value should be NaN, rest shifted
        assert pd.isna(shifted.iloc[0])
        assert shifted.iloc[1] == 1
        assert shifted.iloc[2] == 2
    
    def test_create_lagged_features(self):
        """Test lagged feature creation."""
        df = pd.DataFrame({
            "price": [100, 101, 102, 103, 104],
        })
        
        result = create_lagged_features(df, ["price"], [1, 2])
        
        assert "price_lag1" in result.columns
        assert "price_lag2" in result.columns
        assert pd.isna(result["price_lag1"].iloc[0])
        assert result["price_lag1"].iloc[1] == 100
        assert result["price_lag2"].iloc[2] == 100
    
    def test_create_rolling_features(self):
        """Test rolling feature creation."""
        df = pd.DataFrame({
            "price": [100.0] * 10,
        })
        
        result = create_rolling_features(df, ["price"], [3])
        
        assert "price_roll_mean_3" in result.columns
        assert "price_roll_std_3" in result.columns
        # First few should be NaN due to shift(1)
        assert pd.isna(result["price_roll_mean_3"].iloc[0])
    
    def test_check_for_target_leakage(self):
        """Test target leakage detection."""
        features = pd.DataFrame({
            "feat1": [1, 2, 3, 4, 5],
            "feat2": [1, 2, 3, 4, 5],  # Perfect correlation with target
        })
        target = pd.Series([1, 2, 3, 4, 5])
        
        leaky = check_for_target_leakage(features, target, threshold=0.99)
        
        assert "feat2" in [f[0] for f in leaky]
    
    def test_assert_no_overlap_valid(self):
        """Test no overlap assertion passes."""
        train = pd.DataFrame(index=pd.date_range("2024-01-01", periods=10, freq="1min"))
        test = pd.DataFrame(index=pd.date_range("2024-01-01 00:15", periods=10, freq="1min"))
        
        assert assert_no_overlap(train, test, time_buffer=timedelta(minutes=1))
    
    def test_assert_no_overlap_fails(self):
        """Test no overlap assertion fails with overlap."""
        train = pd.DataFrame(index=pd.date_range("2024-01-01", periods=10, freq="1min"))
        test = pd.DataFrame(index=pd.date_range("2024-01-01 00:05", periods=10, freq="1min"))
        
        with pytest.raises(ValueError):
            assert_no_overlap(train, test)


class TestStability:
    """Tests for stability utilities."""
    
    def test_walk_forward_split(self):
        """Test walk-forward split generation."""
        timestamps = pd.date_range("2024-01-01", periods=1000, freq="1min")
        data = pd.DataFrame({"value": range(1000)}, index=timestamps)
        
        splits = list(walk_forward_split(
            data,
            n_splits=3,
            min_train_size=timedelta(minutes=100),
            test_size=timedelta(minutes=50),
            gap=timedelta(minutes=10),
        ))
        
        assert len(splits) == 3
        
        for train, test in splits:
            assert len(train) > 0
            assert len(test) > 0
            # Train should end before test starts
            assert train.index.max() < test.index.min()
    
    def test_time_series_split(self):
        """Test simple time series split."""
        timestamps = pd.date_range("2024-01-01", periods=100, freq="1min")
        data = pd.DataFrame({"value": range(100)}, index=timestamps)
        
        train, val, test = time_series_split(
            data,
            train_frac=0.6,
            val_frac=0.2,
        )
        
        assert len(train) == 60
        assert len(val) == 20
        assert len(test) == 20
        
        # Should be in order
        assert train.index.max() < val.index.min()
        assert val.index.max() < test.index.min()
    
    def test_regime_split_by_volatility(self):
        """Test volatility regime split."""
        # Create data with varying volatility
        low_vol = np.random.normal(0, 0.001, 100)
        high_vol = np.random.normal(0, 0.01, 100)
        values = np.concatenate([low_vol, high_vol])
        
        timestamps = pd.date_range("2024-01-01", periods=200, freq="1min")
        data = pd.DataFrame({
            "realized_vol_20": np.abs(values),
        }, index=timestamps)
        
        regimes = regime_split_by_volatility(data, n_regimes=2)
        
        assert len(regimes) == 200
        assert set(regimes.unique()) <= {"low_vol", "high_vol"}
    
    def test_regime_split_by_trend(self):
        """Test trend regime split."""
        # Create trending and sideways data
        uptrend = np.cumsum(np.ones(100) * 0.01) + 100
        downtrend = np.cumsum(np.ones(100) * -0.01) + 200
        prices = np.concatenate([uptrend, downtrend])
        
        timestamps = pd.date_range("2024-01-01", periods=200, freq="1min")
        data = pd.DataFrame({
            "mid_price": prices,
        }, index=timestamps)
        
        regimes = regime_split_by_trend(data, window=10)
        
        assert len(regimes) == 200
        assert set(regimes.unique()) <= {"uptrend", "downtrend", "sideways"}
    
    def test_compute_stability_metrics(self):
        """Test stability metrics computation."""
        results = [
            {"sharpe_ratio": 1.0},
            {"sharpe_ratio": 1.2},
            {"sharpe_ratio": 0.8},
        ]
        
        stability = compute_stability_metrics(results, metric_name="sharpe_ratio")
        
        assert "sharpe_ratio_mean" in stability
        assert "sharpe_ratio_std" in stability
        assert stability["sharpe_ratio_mean"] == pytest.approx(1.0)
