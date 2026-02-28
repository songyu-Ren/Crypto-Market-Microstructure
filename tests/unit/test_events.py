"""Tests for data events."""

import pytest
import numpy as np
from datetime import datetime

from crypto_mm_research.data.events import (
    L2BookSnapshotEvent,
    TradeEvent,
    Side,
)


class TestL2BookSnapshotEvent:
    """Tests for L2BookSnapshotEvent."""
    
    def test_basic_creation(self):
        """Test basic event creation."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        bids = np.array([[100.0, 1.0], [99.0, 2.0]])
        asks = np.array([[101.0, 1.5], [102.0, 2.5]])
        
        event = L2BookSnapshotEvent(
            timestamp=timestamp,
            symbol="BTC-USDT",
            bids=bids,
            asks=asks,
        )
        
        assert event.timestamp == timestamp
        assert event.symbol == "BTC-USDT"
        assert event.best_bid == 100.0
        assert event.best_ask == 101.0
    
    def test_mid_price(self):
        """Test mid price calculation."""
        bids = np.array([[100.0, 1.0]])
        asks = np.array([[102.0, 1.0]])
        
        event = L2BookSnapshotEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            bids=bids,
            asks=asks,
        )
        
        assert event.mid_price == 101.0
    
    def test_spread(self):
        """Test spread calculation."""
        bids = np.array([[100.0, 1.0]])
        asks = np.array([[101.0, 1.0]])
        
        event = L2BookSnapshotEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            bids=bids,
            asks=asks,
        )
        
        assert event.spread == 1.0
    
    def test_microprice(self):
        """Test microprice calculation."""
        # Equal sizes -> microprice = mid
        bids = np.array([[100.0, 1.0]])
        asks = np.array([[102.0, 1.0]])
        
        event = L2BookSnapshotEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            bids=bids,
            asks=asks,
        )
        
        assert event.microprice == 101.0  # Mid when sizes equal
    
    def test_microprice_imbalanced(self):
        """Test microprice with imbalanced sizes."""
        # More ask volume -> microprice closer to bid
        bids = np.array([[100.0, 1.0]])
        asks = np.array([[102.0, 3.0]])
        
        event = L2BookSnapshotEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            bids=bids,
            asks=asks,
        )
        
        # Microprice should be closer to bid (100) than mid (101)
        assert event.microprice < 101.0
        assert event.microprice > 100.0
    
    def test_invalid_book_raises(self):
        """Test that crossed book raises error."""
        bids = np.array([[102.0, 1.0]])  # Bid > Ask
        asks = np.array([[100.0, 1.0]])
        
        with pytest.raises(ValueError):
            L2BookSnapshotEvent(
                timestamp=datetime.now(),
                symbol="TEST",
                bids=bids,
                asks=asks,
            )
    
    def test_empty_book(self):
        """Test empty book handling."""
        bids = np.array([]).reshape(0, 2)
        asks = np.array([]).reshape(0, 2)
        
        event = L2BookSnapshotEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            bids=bids,
            asks=asks,
        )
        
        assert event.mid_price == 0.0
        assert event.spread == 0.0


class TestTradeEvent:
    """Tests for TradeEvent."""
    
    def test_basic_creation(self):
        """Test basic trade creation."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        
        event = TradeEvent(
            timestamp=timestamp,
            symbol="BTC-USDT",
            price=100.5,
            size=1.5,
            side=Side.BUY,
            trade_id="t123",
        )
        
        assert event.timestamp == timestamp
        assert event.symbol == "BTC-USDT"
        assert event.price == 100.5
        assert event.size == 1.5
        assert event.side == Side.BUY
        assert event.trade_id == "t123"
    
    def test_invalid_size_raises(self):
        """Test that non-positive size raises error."""
        with pytest.raises(ValueError):
            TradeEvent(
                timestamp=datetime.now(),
                symbol="TEST",
                price=100.0,
                size=0.0,
                side=Side.BUY,
            )
    
    def test_invalid_price_raises(self):
        """Test that non-positive price raises error."""
        with pytest.raises(ValueError):
            TradeEvent(
                timestamp=datetime.now(),
                symbol="TEST",
                price=-100.0,
                size=1.0,
                side=Side.SELL,
            )
    
    def test_sell_side(self):
        """Test sell side trade."""
        event = TradeEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            price=100.0,
            size=2.0,
            side=Side.SELL,
        )
        
        assert event.side == Side.SELL
        assert event.side.value == -1
