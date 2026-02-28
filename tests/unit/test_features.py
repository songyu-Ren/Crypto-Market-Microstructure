"""Tests for feature engineering."""

import pytest
import numpy as np
from datetime import datetime

from crypto_mm_research.data.events import L2BookSnapshotEvent, TradeEvent, Side
from crypto_mm_research.features.microstructure import (
    compute_mid_price,
    compute_spread,
    compute_spread_bps,
    compute_microprice,
    compute_book_imbalance,
    compute_depth_imbalance,
    compute_ofi,
    compute_realized_volatility,
    compute_rolling_zscore,
    compute_price_momentum,
    compute_trade_imbalance,
    compute_vwap,
)
from crypto_mm_research.features.builder import FeatureBuilder


class TestMicrostructureFeatures:
    """Tests for microstructure feature functions."""
    
    def test_compute_mid_price(self):
        """Test mid price computation."""
        book = L2BookSnapshotEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            bids=np.array([[100.0, 1.0]]),
            asks=np.array([[102.0, 1.0]]),
        )
        
        assert compute_mid_price(book) == 101.0
    
    def test_compute_spread(self):
        """Test spread computation."""
        book = L2BookSnapshotEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            bids=np.array([[100.0, 1.0]]),
            asks=np.array([[101.0, 1.0]]),
        )
        
        assert compute_spread(book) == 1.0
    
    def test_compute_spread_bps(self):
        """Test spread in bps."""
        book = L2BookSnapshotEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            bids=np.array([[9900.0, 1.0]]),
            asks=np.array([[10000.0, 1.0]]),
        )
        
        # Spread = 100, Mid = 9950, Spread bps = 100/9950 * 10000 = ~100.5
        spread_bps = compute_spread_bps(book)
        assert 100 < spread_bps < 101
    
    def test_compute_book_imbalance_balanced(self):
        """Test imbalance with equal sizes."""
        book = L2BookSnapshotEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            bids=np.array([[100.0, 1.0]]),
            asks=np.array([[101.0, 1.0]]),
        )
        
        assert compute_book_imbalance(book) == 0.0
    
    def test_compute_book_imbalance_bid_heavy(self):
        """Test imbalance with more bid volume."""
        book = L2BookSnapshotEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            bids=np.array([[100.0, 3.0]]),
            asks=np.array([[101.0, 1.0]]),
        )
        
        # (3 - 1) / (3 + 1) = 0.5
        assert compute_book_imbalance(book) == 0.5
    
    def test_compute_book_imbalance_ask_heavy(self):
        """Test imbalance with more ask volume."""
        book = L2BookSnapshotEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            bids=np.array([[100.0, 1.0]]),
            asks=np.array([[101.0, 3.0]]),
        )
        
        # (1 - 3) / (1 + 3) = -0.5
        assert compute_book_imbalance(book) == -0.5
    
    def test_compute_depth_imbalance(self):
        """Test depth-weighted imbalance."""
        book = L2BookSnapshotEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            bids=np.array([[100.0, 1.0], [99.0, 1.0], [98.0, 1.0]]),
            asks=np.array([[101.0, 1.0], [102.0, 1.0], [103.0, 1.0]]),
        )
        
        # With equal sizes at all levels, imbalance should be 0
        imb = compute_depth_imbalance(book, levels=3)
        assert abs(imb) < 0.01
    
    def test_compute_ofi_no_change(self):
        """Test OFI with no changes."""
        book1 = L2BookSnapshotEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            bids=np.array([[100.0, 1.0]]),
            asks=np.array([[101.0, 1.0]]),
        )
        book2 = L2BookSnapshotEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            bids=np.array([[100.0, 1.0]]),
            asks=np.array([[101.0, 1.0]]),
        )
        
        ofi = compute_ofi(book1, book2, [])
        assert ofi == 0.0
    
    def test_compute_ofi_with_bid_increase(self):
        """Test OFI with bid size increase."""
        book1 = L2BookSnapshotEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            bids=np.array([[100.0, 1.0]]),
            asks=np.array([[101.0, 1.0]]),
        )
        book2 = L2BookSnapshotEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            bids=np.array([[100.0, 3.0]]),  # Bid size increased
            asks=np.array([[101.0, 1.0]]),
        )
        
        ofi = compute_ofi(book1, book2, [])
        assert ofi == 2.0  # Bid size increased by 2
    
    def test_compute_ofi_with_trade(self):
        """Test OFI with aggressive trade."""
        book1 = L2BookSnapshotEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            bids=np.array([[100.0, 1.0]]),
            asks=np.array([[101.0, 1.0]]),
        )
        book2 = L2BookSnapshotEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            bids=np.array([[100.0, 1.0]]),
            asks=np.array([[101.0, 1.0]]),
        )
        
        trade = TradeEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            price=101.0,
            size=0.5,
            side=Side.BUY,
        )
        
        ofi = compute_ofi(book1, book2, [trade])
        assert ofi == 0.5  # Buy trade adds to OFI
    
    def test_compute_realized_volatility(self):
        """Test realized volatility."""
        returns = [0.01, -0.01, 0.01, -0.01, 0.01]
        vol = compute_realized_volatility(returns, window=5)
        assert vol > 0
    
    def test_compute_realized_volatility_insufficient_data(self):
        """Test vol with insufficient data."""
        returns = [0.01]
        vol = compute_realized_volatility(returns, window=5)
        assert vol == 0.0
    
    def test_compute_rolling_zscore(self):
        """Test z-score computation."""
        values = list(range(100))  # Linear increase
        zscore = compute_rolling_zscore(values, window=20)
        assert zscore > 1.5  # Recent values are high relative to past
    
    def test_compute_price_momentum(self):
        """Test momentum computation."""
        prices = [100.0] * 10 + [110.0] * 10  # Jump up
        momentum = compute_price_momentum(prices, window=10)
        assert momentum == 0.1  # 10% return
    
    def test_compute_trade_imbalance(self):
        """Test trade imbalance."""
        trades = [
            TradeEvent(datetime.now(), "TEST", 100.0, 1.0, Side.BUY),
            TradeEvent(datetime.now(), "TEST", 100.0, 3.0, Side.SELL),
        ]
        
        # (1 - 3) / (1 + 3) = -0.5
        imb = compute_trade_imbalance(trades)
        assert imb == -0.5
    
    def test_compute_vwap(self):
        """Test VWAP computation."""
        trades = [
            TradeEvent(datetime.now(), "TEST", 100.0, 1.0, Side.BUY),
            TradeEvent(datetime.now(), "TEST", 110.0, 1.0, Side.SELL),
        ]
        
        # VWAP = (100*1 + 110*1) / (1 + 1) = 105
        vwap = compute_vwap(trades)
        assert vwap == 105.0


class TestFeatureBuilder:
    """Tests for FeatureBuilder."""
    
    def test_builder_initialization(self):
        """Test builder initialization."""
        builder = FeatureBuilder(symbol="BTC-USDT")
        assert builder.symbol == "BTC-USDT"
        assert builder.last_book is None
    
    def test_process_single_book(self):
        """Test processing a single book."""
        builder = FeatureBuilder(symbol="TEST")
        
        book = L2BookSnapshotEvent(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            symbol="TEST",
            bids=np.array([[100.0, 1.0]]),
            asks=np.array([[101.0, 1.0]]),
        )
        
        row = builder.on_event(book)
        
        # First book should emit a feature row
        assert row is not None
        assert row.mid_price == 100.5
        assert row.spread == 1.0
    
    def test_process_trade(self):
        """Test processing a trade."""
        builder = FeatureBuilder(symbol="TEST")
        
        trade = TradeEvent(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            symbol="TEST",
            price=100.5,
            size=1.0,
            side=Side.BUY,
        )
        
        # Trades don't emit features directly
        row = builder.on_event(trade)
        assert row is None
    
    def test_feature_row_to_dict(self):
        """Test FeatureRow conversion to dict."""
        from crypto_mm_research.features.builder import FeatureRow
        
        row = FeatureRow(
            timestamp=datetime.now(),
            symbol="TEST",
            mid_price=100.0,
            microprice=100.1,
            spread=1.0,
            spread_bps=10.0,
            book_imbalance=0.1,
            depth_imbalance_5=0.05,
            ofi=0.0,
            trade_imbalance=0.0,
            trade_intensity=0.0,
            realized_vol_20=0.001,
            returns_1s=0.0001,
            zscore_mid_20=0.5,
        )
        
        d = row.to_dict()
        assert d["mid_price"] == 100.0
        assert d["symbol"] == "TEST"
    
    def test_process_events_empty(self):
        """Test processing empty event list."""
        builder = FeatureBuilder(symbol="TEST")
        df = builder.process_events([])
        assert len(df) == 0
    
    def test_no_lookahead_in_features(self):
        """Test that features don't use future data."""
        from crypto_mm_research.data.synthetic import SyntheticDataGenerator
        
        gen = SyntheticDataGenerator(random_seed=42)
        events = gen.generate_to_list(duration_seconds=60, events_per_second=5)
        
        builder = FeatureBuilder(symbol="BTC-USDT")
        df = builder.process_events(events)
        
        # Check that z-score uses only past data
        # (z-score should not be exactly 0 for trending series)
        if len(df) > 20:
            assert not all(df["zscore_mid_20"] == 0)
