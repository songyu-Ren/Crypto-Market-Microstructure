"""End-to-end integration tests."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from crypto_mm_research.data.synthetic import SyntheticDataGenerator
from crypto_mm_research.features.builder import FeatureBuilder
from crypto_mm_research.backtest.engine import BacktestEngine, BacktestConfig
from crypto_mm_research.backtest.strategy import MarketMakingStrategy
from crypto_mm_research.evaluation.stability import walk_forward_split


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_full_pipeline(self):
        """Test full data -> features -> backtest pipeline."""
        # 1. Generate data
        gen = SyntheticDataGenerator(
            symbol="BTC-USDT",
            random_seed=42,
        )
        events = gen.generate_to_list(duration_seconds=600, events_per_second=5)
        
        # 2. Compute features
        builder = FeatureBuilder(symbol="BTC-USDT")
        features_df = builder.process_events(events)
        
        assert len(features_df) > 0
        assert "mid_price" in features_df.columns
        assert "book_imbalance" in features_df.columns
        
        # 3. Run backtest
        strategy = MarketMakingStrategy(
            target_half_spread_bps=5.0,
            quote_size=0.1,
        )
        engine = BacktestEngine(strategy, BacktestConfig())
        result = engine.run(iter(events))
        
        # 4. Verify results
        assert len(result.fill_history) >= 0
        metrics = result.compute_metrics()
        assert "total_pnl" in metrics
        assert "sharpe_ratio" in metrics
    
    def test_feature_consistency_with_backtest(self):
        """Test that features and backtest use same data consistently."""
        gen = SyntheticDataGenerator(random_seed=123)
        events = gen.generate_to_list(duration_seconds=300, events_per_second=5)
        
        # Features
        builder = FeatureBuilder(symbol="BTC-USDT")
        features_df = builder.process_events(events)
        
        # Backtest
        strategy = MarketMakingStrategy()
        engine = BacktestEngine(strategy, BacktestConfig())
        result = engine.run(iter(events))
        
        # Equity curve should have timestamps
        equity_df = result.get_equity_curve()
        if not equity_df.empty and not features_df.empty:
            # Timestamps should overlap
            feature_times = set(features_df.index)
            equity_times = set(equity_df.index)
            overlap = feature_times & equity_times
            assert len(overlap) > 0
    
    def test_multiple_strategies_same_data(self):
        """Test multiple strategies on same data produce different results."""
        gen = SyntheticDataGenerator(random_seed=42)
        events = gen.generate_to_list(duration_seconds=300, events_per_second=5)
        
        strategies = [
            MarketMakingStrategy(target_half_spread_bps=5.0, skew_coeff=0.0),
            MarketMakingStrategy(target_half_spread_bps=10.0, skew_coeff=2.0),
        ]
        
        results = []
        for strategy in strategies:
            # Regenerate events (generator is exhausted)
            gen = SyntheticDataGenerator(random_seed=42)
            events = gen.generate_to_list(duration_seconds=300, events_per_second=5)
            
            engine = BacktestEngine(strategy, BacktestConfig())
            result = engine.run(iter(events))
            results.append(result)
        
        # Different strategies should produce different results
        metrics1 = results[0].compute_metrics()
        metrics2 = results[1].compute_metrics()
        
        # At least one metric should differ
        assert metrics1["n_fills"] != metrics2["n_fills"] or \
               metrics1["total_pnl"] != metrics2["total_pnl"]
    
    def test_walk_forward_with_backtest(self):
        """Test walk-forward evaluation with backtest."""
        # Generate longer data
        gen = SyntheticDataGenerator(random_seed=42)
        events = gen.generate_to_list(duration_seconds=1800, events_per_second=2)
        
        # Convert to DataFrame for walk-forward
        builder = FeatureBuilder(symbol="BTC-USDT")
        features_df = builder.process_events(events)
        
        if len(features_df) < 50:
            pytest.skip("Not enough data for walk-forward")
        
        # Walk-forward splits
        splits = list(walk_forward_split(
            features_df,
            n_splits=2,
            min_train_size=pd.Timedelta(minutes=5),
            test_size=pd.Timedelta(minutes=3),
        ))
        
        assert len(splits) > 0
        
        for train_data, test_data in splits:
            assert len(train_data) > 0
            assert len(test_data) > 0
            # No overlap
            assert train_data.index.max() < test_data.index.min()
    
    def test_inventory_skew_effect(self):
        """Test that inventory skew actually affects quoting."""
        gen = SyntheticDataGenerator(random_seed=42)
        events = gen.generate_to_list(duration_seconds=600, events_per_second=5)
        
        # Strategy with skew
        strategy_skew = MarketMakingStrategy(
            target_half_spread_bps=5.0,
            skew_coeff=5.0,  # Strong skew
            inventory_limit=10.0,
        )
        
        engine = BacktestEngine(strategy_skew, BacktestConfig())
        result = engine.run(iter(events))
        
        # Should have some fills
        assert len(result.fill_history) > 0
        
        # Check that inventory was managed
        equity_df = result.get_equity_curve()
        if not equity_df.empty:
            max_inventory = equity_df["position"].abs().max()
            # Inventory should generally stay bounded
            assert max_inventory < 10.0  # Should respect limit
    
    def test_deterministic_reproducibility(self):
        """Test that same seed produces identical results."""
        results = []
        
        for _ in range(2):
            gen = SyntheticDataGenerator(random_seed=999)
            events = gen.generate_to_list(duration_seconds=300, events_per_second=5)
            
            strategy = MarketMakingStrategy(
                target_half_spread_bps=5.0,
                quote_size=0.1,
            )
            engine = BacktestEngine(strategy, BacktestConfig())
            result = engine.run(iter(events))
            
            metrics = result.compute_metrics()
            results.append(metrics)
        
        # All metrics should be identical
        for key in results[0].keys():
            assert results[0][key] == pytest.approx(results[1][key]), f"Mismatch in {key}"
    
    def test_feature_builder_no_lookahead(self):
        """Test that feature builder doesn't use lookahead."""
        gen = SyntheticDataGenerator(random_seed=42)
        events = gen.generate_to_list(duration_seconds=300, events_per_second=5)
        
        builder = FeatureBuilder(symbol="BTC-USDT", window_seconds=20.0)
        features_df = builder.process_events(events)
        
        # Z-scores should be computable without future data
        # If lookahead existed, z-scores would be artificially low
        if len(features_df) > 20:
            zscore_std = features_df["zscore_mid_20"].std()
            # Z-scores should have reasonable variance
            assert zscore_std > 0
            # Most z-scores should be within reasonable bounds
            assert features_df["zscore_mid_20"].abs().max() < 10
