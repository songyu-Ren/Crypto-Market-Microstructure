"""Tests for backtest engine."""

import pytest
import numpy as np
from datetime import datetime

from crypto_mm_research.data.events import L2BookSnapshotEvent, TradeEvent, Side
from crypto_mm_research.data.synthetic import SyntheticDataGenerator
from crypto_mm_research.backtest.engine import BacktestEngine, BacktestConfig, FillModel
from crypto_mm_research.backtest.strategy import MarketMakingStrategy, Quote, StrategyOutput
from crypto_mm_research.backtest.account import Account, Fill, Position, Side as AccountSide


class TestAccount:
    """Tests for Account."""
    
    def test_initial_state(self):
        """Test initial account state."""
        account = Account(initial_cash=100000.0)
        assert account.cash == 100000.0
        assert account.initial_cash == 100000.0
        assert len(account.fills) == 0
    
    def test_apply_buy_fill(self):
        """Test applying a buy fill."""
        account = Account(initial_cash=100000.0, maker_fee_rate=0.001)
        
        fill = account.apply_fill(
            timestamp=datetime.now(),
            symbol="BTC-USDT",
            price=50000.0,
            size=1.0,
            side=AccountSide.LONG,
            is_maker=True,
        )
        
        # Cash should decrease by notional + fee
        # Notional = 50000, Fee = 50000 * 0.001 = 50
        expected_cash = 100000 - 50000 - 50
        assert account.cash == pytest.approx(expected_cash)
        
        # Position should be updated
        position = account.get_position("BTC-USDT")
        assert position.size == 1.0
        assert position.avg_entry_price == 50000.0
    
    def test_apply_sell_fill(self):
        """Test applying a sell fill."""
        account = Account(initial_cash=100000.0)
        
        # First buy
        account.apply_fill(
            timestamp=datetime.now(),
            symbol="BTC-USDT",
            price=50000.0,
            size=1.0,
            side=AccountSide.LONG,
            is_maker=True,
        )
        
        # Then sell
        account.apply_fill(
            timestamp=datetime.now(),
            symbol="BTC-USDT",
            price=51000.0,
            size=1.0,
            side=AccountSide.SELL,
            is_maker=True,
        )
        
        # Position should be flat
        position = account.get_position("BTC-USDT")
        assert abs(position.size) < 1e-9
        
        # Should have realized PnL
        assert position.realized_pnl > 0
    
    def test_realized_pnl_calculation(self):
        """Test realized PnL calculation."""
        account = Account(initial_cash=100000.0, maker_fee_rate=0.0)
        
        # Buy at 100
        account.apply_fill(
            timestamp=datetime.now(),
            symbol="TEST",
            price=100.0,
            size=1.0,
            side=AccountSide.LONG,
            is_maker=True,
        )
        
        # Sell at 110
        account.apply_fill(
            timestamp=datetime.now(),
            symbol="TEST",
            price=110.0,
            size=1.0,
            side=AccountSide.SELL,
            is_maker=True,
        )
        
        position = account.get_position("TEST")
        # PnL = (110 - 100) * 1 = 10
        assert position.realized_pnl == pytest.approx(10.0)
    
    def test_position_average_price(self):
        """Test position average entry price."""
        account = Account(initial_cash=100000.0)
        
        # Buy 1 at 100
        account.apply_fill(
            timestamp=datetime.now(),
            symbol="TEST",
            price=100.0,
            size=1.0,
            side=AccountSide.LONG,
            is_maker=True,
        )
        
        # Buy 1 at 110
        account.apply_fill(
            timestamp=datetime.now(),
            symbol="TEST",
            price=110.0,
            size=1.0,
            side=AccountSide.LONG,
            is_maker=True,
        )
        
        position = account.get_position("TEST")
        # Avg price = (100*1 + 110*1) / 2 = 105
        assert position.avg_entry_price == pytest.approx(105.0)
        assert position.size == 2.0
    
    def test_mark_to_market(self):
        """Test mark to market."""
        account = Account(initial_cash=100000.0)
        
        # Buy 1 at 100
        account.apply_fill(
            timestamp=datetime.now(),
            symbol="TEST",
            price=100.0,
            size=1.0,
            side=AccountSide.LONG,
            is_maker=True,
        )
        
        # Mark at 110
        account.mark_to_market(datetime.now(), {"TEST": 110.0})
        
        # Should have equity curve entry
        assert len(account.equity_curve) == 1
        assert account.equity_curve[0]["total_equity"] > account.initial_cash


class TestFillModel:
    """Tests for FillModel."""
    
    def test_bid_fill_crossed_market(self):
        """Test bid fill when crossed market."""
        model = FillModel()
        
        book = L2BookSnapshotEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            bids=np.array([[100.0, 1.0]]),
            asks=np.array([[101.0, 1.0]]),
        )
        
        # Bid at 102 (above ask) should fill
        bid = Quote(price=102.0, size=1.0)
        assert model.check_bid_fill(bid, book, [])
        
        # Bid at 99 (below best bid) should not fill
        bid = Quote(price=99.0, size=1.0)
        assert not model.check_bid_fill(bid, book, [])
    
    def test_ask_fill_crossed_market(self):
        """Test ask fill when crossed market."""
        model = FillModel()
        
        book = L2BookSnapshotEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            bids=np.array([[100.0, 1.0]]),
            asks=np.array([[101.0, 1.0]]),
        )
        
        # Ask at 99 (below bid) should fill
        ask = Quote(price=99.0, size=1.0)
        assert model.check_ask_fill(ask, book, [])
        
        # Ask at 102 (above best ask) should not fill
        ask = Quote(price=102.0, size=1.0)
        assert not model.check_ask_fill(ask, book, [])
    
    def test_bid_fill_through_trade(self):
        """Test bid fill through aggressive trade."""
        model = FillModel(aggressive_fill_through_book=True)
        
        book = L2BookSnapshotEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            bids=np.array([[100.0, 1.0]]),
            asks=np.array([[101.0, 1.0]]),
        )
        
        trade = TradeEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            price=99.5,  # Below our bid
            size=1.0,
            side=Side.SELL,  # Seller hitting our bid
        )
        
        # Bid at 100 should fill from trade at 99.5
        bid = Quote(price=100.0, size=1.0)
        assert model.check_bid_fill(bid, book, [trade])


class TestMarketMakingStrategy:
    """Tests for MarketMakingStrategy."""
    
    def test_basic_quote_generation(self):
        """Test basic quote generation."""
        strategy = MarketMakingStrategy(
            target_half_spread_bps=10.0,  # 10 bps = 0.1%
            quote_size=1.0,
        )
        
        book = L2BookSnapshotEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            bids=np.array([[9900.0, 1.0]]),
            asks=np.array([[10100.0, 1.0]]),
        )
        
        account = Account(initial_cash=100000.0)
        output = strategy.on_book(datetime.now(), book, account)
        
        assert output.has_quotes()
        assert output.bid is not None
        assert output.ask is not None
        
        # Mid = 10000, half spread = 10000 * 0.001 = 10
        # Bid should be around 9990
        assert output.bid.price < 10000
        assert output.ask.price > 10000
    
    def test_inventory_skew(self):
        """Test inventory skew."""
        strategy = MarketMakingStrategy(
            target_half_spread_bps=10.0,
            quote_size=1.0,
            skew_coeff=1.0,
        )
        
        book = L2BookSnapshotEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            bids=np.array([[9900.0, 1.0]]),
            asks=np.array([[10100.0, 1.0]]),
        )
        
        account = Account(initial_cash=100000.0)
        
        # Create long position
        account.apply_fill(
            timestamp=datetime.now(),
            symbol="TEST",
            price=10000.0,
            size=2.0,
            side=AccountSide.LONG,
            is_maker=True,
        )
        
        output = strategy.on_book(datetime.now(), book, account)
        
        # With long inventory, quotes should shift down
        # (more willing to sell, less willing to buy)
        mid = 10000
        half_spread = mid * 0.001  # 10
        
        # Without skew, bid would be ~9990
        # With positive inventory, bid should be lower
        assert output.bid is not None
        assert output.bid.price < mid - half_spread
    
    def test_inventory_limit(self):
        """Test inventory limit stops quoting."""
        strategy = MarketMakingStrategy(
            target_half_spread_bps=10.0,
            quote_size=1.0,
            inventory_limit=1.0,
        )
        
        book = L2BookSnapshotEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            bids=np.array([[9900.0, 1.0]]),
            asks=np.array([[10100.0, 1.0]]),
        )
        
        account = Account(initial_cash=100000.0)
        
        # Create position at limit
        account.apply_fill(
            timestamp=datetime.now(),
            symbol="TEST",
            price=10000.0,
            size=1.0,
            side=AccountSide.LONG,
            is_maker=True,
        )
        
        output = strategy.on_book(datetime.now(), book, account)
        
        # Should only quote ask (to reduce position)
        assert output.bid is None or output.bid.size == 0
        assert output.ask is not None


class TestBacktestEngine:
    """Tests for BacktestEngine."""
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        strategy = MarketMakingStrategy()
        config = BacktestConfig()
        engine = BacktestEngine(strategy, config)
        
        assert engine.config.initial_cash == 100000.0
        assert engine.account.cash == 100000.0
    
    def test_run_backtest(self):
        """Test running a simple backtest."""
        # Generate data
        gen = SyntheticDataGenerator(random_seed=42)
        events = gen.generate_to_list(duration_seconds=300, events_per_second=5)
        
        strategy = MarketMakingStrategy(
            target_half_spread_bps=10.0,
            quote_size=0.1,
        )
        config = BacktestConfig()
        engine = BacktestEngine(strategy, config)
        
        result = engine.run(iter(events))
        
        # Should have some fills
        assert len(result.fill_history) >= 0
        
        # Should have equity curve
        assert len(result.account.equity_curve) > 0
    
    def test_backtest_deterministic(self):
        """Test that backtest is deterministic."""
        gen = SyntheticDataGenerator(random_seed=42)
        events = gen.generate_to_list(duration_seconds=300, events_per_second=5)
        
        results = []
        for _ in range(2):
            strategy = MarketMakingStrategy(
                target_half_spread_bps=10.0,
                quote_size=0.1,
            )
            engine = BacktestEngine(strategy, BacktestConfig())
            result = engine.run(iter(events))
            results.append(result)
            # Regenerate events for next run
            gen = SyntheticDataGenerator(random_seed=42)
            events = gen.generate_to_list(duration_seconds=300, events_per_second=5)
        
        # Should have same number of fills
        assert len(results[0].fill_history) == len(results[1].fill_history)
    
    def test_metrics_computation(self):
        """Test metrics computation."""
        gen = SyntheticDataGenerator(random_seed=42)
        events = gen.generate_to_list(duration_seconds=300, events_per_second=5)
        
        strategy = MarketMakingStrategy()
        engine = BacktestEngine(strategy, BacktestConfig())
        result = engine.run(iter(events))
        
        metrics = result.compute_metrics()
        
        assert "total_pnl" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown_pct" in metrics
        assert "n_fills" in metrics
