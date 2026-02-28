"""Tests for synthetic data generator."""

import pytest
import numpy as np
from datetime import datetime

from crypto_mm_research.data.synthetic import SyntheticDataGenerator
from crypto_mm_research.data.events import L2BookSnapshotEvent, TradeEvent


class TestSyntheticDataGenerator:
    """Tests for SyntheticDataGenerator."""
    
    def test_initialization(self):
        """Test generator initialization."""
        gen = SyntheticDataGenerator(
            symbol="ETH-USDT",
            start_price=3000.0,
            random_seed=123,
        )
        
        assert gen.symbol == "ETH-USDT"
        assert gen.start_price == 3000.0
        assert gen.current_mid == 3000.0
    
    def test_deterministic_generation(self):
        """Test that same seed produces same data."""
        gen1 = SyntheticDataGenerator(random_seed=42)
        gen2 = SyntheticDataGenerator(random_seed=42)
        
        events1 = gen1.generate_to_list(duration_seconds=60, events_per_second=5)
        events2 = gen2.generate_to_list(duration_seconds=60, events_per_second=5)
        
        assert len(events1) == len(events2)
        
        # Check first few events match
        for e1, e2 in zip(events1[:5], events2[:5]):
            if isinstance(e1, L2BookSnapshotEvent):
                assert e1.best_bid == e2.best_bid
                assert e1.best_ask == e2.best_ask
    
    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different data."""
        gen1 = SyntheticDataGenerator(random_seed=42)
        gen2 = SyntheticDataGenerator(random_seed=43)
        
        events1 = gen1.generate_to_list(duration_seconds=60, events_per_second=5)
        events2 = gen2.generate_to_list(duration_seconds=60, events_per_second=5)
        
        # At least some events should differ
        prices1 = [e.best_bid for e in events1 if isinstance(e, L2BookSnapshotEvent)]
        prices2 = [e.best_bid for e in events2 if isinstance(e, L2BookSnapshotEvent)]
        
        assert prices1 != prices2
    
    def test_generates_both_event_types(self):
        """Test that both books and trades are generated."""
        gen = SyntheticDataGenerator(random_seed=42)
        events = gen.generate_to_list(duration_seconds=300, events_per_second=10)
        
        books = [e for e in events if isinstance(e, L2BookSnapshotEvent)]
        trades = [e for e in events if isinstance(e, TradeEvent)]
        
        assert len(books) > 0
        assert len(trades) > 0
        assert len(books) > len(trades)  # More books than trades
    
    def test_book_structure(self):
        """Test that books have correct structure."""
        gen = SyntheticDataGenerator(book_levels=5, random_seed=42)
        events = gen.generate_to_list(duration_seconds=10, events_per_second=5)
        
        books = [e for e in events if isinstance(e, L2BookSnapshotEvent)]
        
        for book in books:
            assert book.bids.shape[0] == 5
            assert book.asks.shape[0] == 5
            assert book.bids.shape[1] == 2
            assert book.asks.shape[1] == 2
            
            # Bids sorted descending
            for i in range(1, len(book.bids)):
                assert book.bids[i, 0] < book.bids[i-1, 0]
            
            # Asks sorted ascending
            for i in range(1, len(book.asks)):
                assert book.asks[i, 0] > book.asks[i-1, 0]
    
    def test_prices_respect_tick_size(self):
        """Test that prices are rounded to tick size."""
        gen = SyntheticDataGenerator(
            tick_size=0.5,
            start_price=100.0,
            random_seed=42,
        )
        events = gen.generate_to_list(duration_seconds=60, events_per_second=5)
        
        for event in events:
            if isinstance(event, L2BookSnapshotEvent):
                for price, _ in event.bids:
                    assert price % 0.5 == 0
                for price, _ in event.asks:
                    assert price % 0.5 == 0
    
    def test_sizes_respect_lot_size(self):
        """Test that sizes are rounded to lot size."""
        gen = SyntheticDataGenerator(
            lot_size=0.01,
            random_seed=42,
        )
        events = gen.generate_to_list(duration_seconds=60, events_per_second=5)
        
        for event in events:
            if isinstance(event, L2BookSnapshotEvent):
                for _, size in event.bids:
                    assert size % 0.01 < 1e-9 or size == 0.01
                for _, size in event.asks:
                    assert size % 0.01 < 1e-9 or size == 0.01
    
    def test_spread_respects_target(self):
        """Test that spread is around target."""
        gen = SyntheticDataGenerator(
            target_spread_bps=10.0,  # 10 bps
            start_price=10000.0,
            random_seed=42,
        )
        events = gen.generate_to_list(duration_seconds=300, events_per_second=10)
        
        books = [e for e in events if isinstance(e, L2BookSnapshotEvent)]
        spreads_bps = [10000 * b.spread / b.mid_price for b in books]
        
        # Average spread should be close to target
        avg_spread = np.mean(spreads_bps)
        assert 5 < avg_spread < 20  # Within reasonable range
    
    def test_price_random_walk(self):
        """Test that price follows random walk."""
        gen = SyntheticDataGenerator(
            start_price=50000.0,
            volatility_annual=0.8,
            random_seed=42,
        )
        events = gen.generate_to_list(duration_seconds=3600, events_per_second=1)
        
        books = [e for e in events if isinstance(e, L2BookSnapshotEvent)]
        prices = [b.mid_price for b in books]
        
        # Price should have moved from start
        assert max(prices) != min(prices)
        
        # Price should stay within reasonable bounds
        assert all(40000 < p < 60000 for p in prices)
    
    def test_trade_properties(self):
        """Test trade event properties."""
        gen = SyntheticDataGenerator(random_seed=42)
        events = gen.generate_to_list(duration_seconds=300, events_per_second=10)
        
        trades = [e for e in events if isinstance(e, TradeEvent)]
        
        for trade in trades:
            assert trade.size > 0
            assert trade.price > 0
            assert trade.symbol == gen.symbol
    
    def test_chronological_order(self):
        """Test that events are in chronological order."""
        gen = SyntheticDataGenerator(random_seed=42)
        events = gen.generate_to_list(duration_seconds=60, events_per_second=10)
        
        for i in range(1, len(events)):
            assert events[i].timestamp >= events[i-1].timestamp
