"""Event-driven backtest engine for market making strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Iterator
import pandas as pd

from crypto_mm_research.data.events import L2BookSnapshotEvent, TradeEvent, Event
from crypto_mm_research.backtest.account import Account, Fill, Side
from crypto_mm_research.backtest.strategy import Strategy, StrategyOutput, Quote


@dataclass
class FillModel:
    """Fill model configuration."""
    
    # Fee rates (positive = fee, negative = rebate)
    maker_fee_rate: float = 0.0002  # 2 bps maker fee
    taker_fee_rate: float = 0.0005  # 5 bps taker fee
    
    # Fill conditions
    aggressive_fill_through_book: bool = True  # Fill if trade through our price
    
    def check_bid_fill(
        self,
        bid: Quote,
        book: L2BookSnapshotEvent,
        trades: List[TradeEvent],
    ) -> bool:
        """Check if bid would be filled.
        
        Fill conditions:
        1. Our bid >= best_ask (crossed market)
        2. Trade prints at or through our price
        """
        # Check crossed market
        if bid.price >= book.best_ask:
            return True
        
        # Check trade through
        if self.aggressive_fill_through_book:
            for trade in trades:
                if trade.side == Side.SELL:  # Seller hitting our bid
                    if trade.price <= bid.price:
                        return True
        
        return False
    
    def check_ask_fill(
        self,
        ask: Quote,
        book: L2BookSnapshotEvent,
        trades: List[TradeEvent],
    ) -> bool:
        """Check if ask would be filled.
        
        Fill conditions:
        1. Our ask <= best_bid (crossed market)
        2. Trade prints at or through our price
        """
        # Check crossed market
        if ask.price <= book.best_bid:
            return True
        
        # Check trade through
        if self.aggressive_fill_through_book:
            for trade in trades:
                if trade.side == Side.LONG:  # Buyer lifting our ask
                    if trade.price >= ask.price:
                        return True
        
        return False


@dataclass
class BacktestConfig:
    """Configuration for backtest run."""
    
    symbol: str = "BTC-USDT"
    initial_cash: float = 100000.0
    
    # Fill model
    maker_fee_rate: float = 0.0002
    taker_fee_rate: float = 0.0005
    
    # Recording
    record_interval_seconds: float = 1.0


class BacktestEngine:
    """Event-driven backtest engine.
    
    Processes events in chronological order:
    1. On each book snapshot: strategy generates quotes
    2. On subsequent events: check for fills
    3. Record equity at intervals
    """
    
    def __init__(
        self,
        strategy: Strategy,
        config: Optional[BacktestConfig] = None,
    ) -> None:
        """Initialize backtest engine.
        
        Args:
            strategy: Trading strategy to backtest.
            config: Backtest configuration.
        """
        self.strategy = strategy
        self.config = config or BacktestConfig()
        
        # Initialize account
        self.account = Account(
            initial_cash=self.config.initial_cash,
            maker_fee_rate=self.config.maker_fee_rate,
            taker_fee_rate=self.config.taker_fee_rate,
        )
        
        # Fill model
        self.fill_model = FillModel(
            maker_fee_rate=self.config.maker_fee_rate,
            taker_fee_rate=self.config.taker_fee_rate,
        )
        
        # State
        self.current_book: Optional[L2BookSnapshotEvent] = None
        self.current_quotes: StrategyOutput = StrategyOutput()
        self.pending_trades: List[TradeEvent] = []
        self.last_record_time: Optional[datetime] = None
        
        # Results
        self.fill_history: List[Fill] = []
    
    def reset(self) -> None:
        """Reset engine for new backtest."""
        self.account = Account(
            initial_cash=self.config.initial_cash,
            maker_fee_rate=self.config.maker_fee_rate,
            taker_fee_rate=self.config.taker_fee_rate,
        )
        self.current_book = None
        self.current_quotes = StrategyOutput()
        self.pending_trades = []
        self.last_record_time = None
        self.fill_history = []
        self.strategy.reset()
    
    def _check_fills(
        self,
        timestamp: datetime,
        book: L2BookSnapshotEvent,
        trades: List[TradeEvent],
    ) -> None:
        """Check if current quotes would be filled."""
        symbol = book.symbol
        
        # Check bid fill
        if self.current_quotes.bid is not None:
            if self.fill_model.check_bid_fill(self.current_quotes.bid, book, trades):
                fill = self.account.apply_fill(
                    timestamp=timestamp,
                    symbol=symbol,
                    price=self.current_quotes.bid.price,
                    size=self.current_quotes.bid.size,
                    side=Side.LONG,
                    is_maker=True,
                )
                self.fill_history.append(fill)
        
        # Check ask fill
        if self.current_quotes.ask is not None:
            if self.fill_model.check_ask_fill(self.current_quotes.ask, book, trades):
                fill = self.account.apply_fill(
                    timestamp=timestamp,
                    symbol=symbol,
                    price=self.current_quotes.ask.price,
                    size=self.current_quotes.ask.size,
                    side=Side.SELL,
                    is_maker=True,
                )
                self.fill_history.append(fill)
    
    def _record_equity(self, timestamp: datetime) -> None:
        """Record equity at intervals."""
        if self.current_book is None:
            return
        
        if self.last_record_time is None:
            self.last_record_time = timestamp
            return
        
        from datetime import timedelta
        interval = timedelta(seconds=self.config.record_interval_seconds)
        
        if timestamp - self.last_record_time >= interval:
            prices = {self.current_book.symbol: self.current_book.mid_price}
            self.account.mark_to_market(timestamp, prices)
            self.last_record_time = timestamp
    
    def process_event(self, event: Event) -> None:
        """Process a single event."""
        if isinstance(event, L2BookSnapshotEvent):
            self._process_book(event)
        elif isinstance(event, TradeEvent):
            self._process_trade(event)
    
    def _process_book(self, book: L2BookSnapshotEvent) -> None:
        """Process book snapshot."""
        # Check fills from previous state
        if self.current_book is not None:
            self._check_fills(book.timestamp, book, self.pending_trades)
        
        # Clear pending trades
        self.pending_trades = []
        
        # Update book
        self.current_book = book
        
        # Get new quotes from strategy
        self.current_quotes = self.strategy.on_book(
            book.timestamp, book, self.account
        )
        
        # Record equity
        self._record_equity(book.timestamp)
    
    def _process_trade(self, trade: TradeEvent) -> None:
        """Process trade event."""
        self.pending_trades.append(trade)
        
        # Notify strategy
        self.strategy.on_trade(trade.timestamp, trade, self.account)
    
    def run(self, events: Iterator[Event]) -> "BacktestResult":
        """Run backtest on event stream.
        
        Args:
            events: Iterator of market events.
        
        Returns:
            BacktestResult with performance metrics.
        """
        self.reset()
        
        for event in events:
            self.process_event(event)
        
        # Final mark to market
        if self.current_book is not None:
            prices = {self.current_book.symbol: self.current_book.mid_price}
            self.account.mark_to_market(self.current_book.timestamp, prices)
        
        return BacktestResult(
            account=self.account,
            fill_history=self.fill_history,
            config=self.config,
        )


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    
    account: Account
    fill_history: List[Fill]
    config: BacktestConfig
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.account.equity_curve:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.account.equity_curve)
        if not df.empty:
            df.set_index("timestamp", inplace=True)
        return df
    
    def get_fills_df(self) -> pd.DataFrame:
        """Get fills as DataFrame."""
        if not self.fill_history:
            return pd.DataFrame()
        
        data = []
        for fill in self.fill_history:
            data.append({
                "timestamp": fill.timestamp,
                "symbol": fill.symbol,
                "price": fill.price,
                "size": fill.size,
                "side": "buy" if fill.side == Side.LONG else "sell",
                "fee": fill.fee,
                "is_maker": fill.is_maker,
            })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index("timestamp", inplace=True)
        return df
    
    def compute_metrics(self) -> Dict:
        """Compute performance metrics."""
        from crypto_mm_research.backtest.metrics import compute_metrics
        return compute_metrics(self)
