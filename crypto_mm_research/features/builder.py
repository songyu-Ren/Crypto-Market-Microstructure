"""FeatureBuilder for sequential feature computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from crypto_mm_research.data.events import L2BookSnapshotEvent, TradeEvent, Event
from crypto_mm_research.features import microstructure as ms


@dataclass
class FeatureRow:
    """A single row of computed features at time t.
    
    All features are computed using only information available up to timestamp.
    """
    
    timestamp: datetime
    symbol: str
    
    # Price features
    mid_price: float
    microprice: float
    spread: float
    spread_bps: float
    
    # Book features
    book_imbalance: float
    depth_imbalance_5: float
    
    # Flow features
    ofi: float
    trade_imbalance: float
    trade_intensity: float
    
    # Volatility features
    realized_vol_20: float
    returns_1s: float
    zscore_mid_20: float
    
    # Extra features dict for extensibility
    extra: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, float | str | datetime]:
        """Convert to dictionary."""
        result = {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "mid_price": self.mid_price,
            "microprice": self.microprice,
            "spread": self.spread,
            "spread_bps": self.spread_bps,
            "book_imbalance": self.book_imbalance,
            "depth_imbalance_5": self.depth_imbalance_5,
            "ofi": self.ofi,
            "trade_imbalance": self.trade_imbalance,
            "trade_intensity": self.trade_intensity,
            "realized_vol_20": self.realized_vol_20,
            "returns_1s": self.returns_1s,
            "zscore_mid_20": self.zscore_mid_20,
        }
        result.update(self.extra)
        return result


class FeatureBuilder:
    """Build features sequentially from event stream.
    
    This class processes events in chronological order and maintains
    state for rolling computations. It ensures no lookahead bias by
    only using information available at or before each timestamp.
    
    Example:
        builder = FeatureBuilder(window_seconds=20.0)
        for event in events:
            row = builder.on_event(event)
            if row is not None:
                features.append(row)
    """
    
    def __init__(
        self,
        symbol: str = "BTC-USDT",
        window_seconds: float = 20.0,
        max_history_seconds: float = 300.0,
    ) -> None:
        """Initialize FeatureBuilder.
        
        Args:
            symbol: Symbol to process.
            window_seconds: Default rolling window for features.
            max_history_seconds: Maximum history to retain in memory.
        """
        self.symbol = symbol
        self.window_seconds = window_seconds
        self.max_history_seconds = max_history_seconds
        
        # State
        self.last_book: Optional[L2BookSnapshotEvent] = None
        self.prev_book: Optional[L2BookSnapshotEvent] = None
        self.mid_prices: List[tuple[datetime, float]] = []
        self.returns: List[float] = []
        self.trades: List[TradeEvent] = []
        self.last_feature_time: Optional[datetime] = None
        self.min_feature_interval = timedelta(seconds=1.0)
    
    def _cleanup_old_data(self, current_time: datetime) -> None:
        """Remove data older than max_history_seconds."""
        cutoff = current_time - timedelta(seconds=self.max_history_seconds)
        
        self.mid_prices = [
            (t, p) for t, p in self.mid_prices if t >= cutoff
        ]
        self.trades = [t for t in self.trades if t.timestamp >= cutoff]
    
    def _compute_returns(self) -> None:
        """Compute returns from mid price history."""
        if len(self.mid_prices) < 2:
            self.returns = []
            return
        
        prices = [p for _, p in self.mid_prices]
        self.returns = []
        for i in range(1, len(prices)):
            if prices[i - 1] != 0:
                self.returns.append((prices[i] - prices[i - 1]) / prices[i - 1])
    
    def _get_recent_trades(
        self, current_time: datetime, window_seconds: float
    ) -> List[TradeEvent]:
        """Get trades within window."""
        cutoff = current_time - timedelta(seconds=window_seconds)
        return [t for t in self.trades if t.timestamp >= cutoff]
    
    def _should_emit_feature(self, timestamp: datetime) -> bool:
        """Check if enough time has passed to emit new feature row."""
        if self.last_feature_time is None:
            return True
        return timestamp - self.last_feature_time >= self.min_feature_interval
    
    def on_event(self, event: Event) -> Optional[FeatureRow]:
        """Process a single event and optionally return a feature row.
        
        Args:
            event: L2BookSnapshotEvent or TradeEvent.
        
        Returns:
            FeatureRow if enough time has passed, None otherwise.
        """
        if isinstance(event, L2BookSnapshotEvent):
            return self._on_book(event)
        elif isinstance(event, TradeEvent):
            return self._on_trade(event)
        return None
    
    def _on_book(self, book: L2BookSnapshotEvent) -> Optional[FeatureRow]:
        """Process book snapshot."""
        if book.symbol != self.symbol:
            return None
        
        # Update state
        self.prev_book = self.last_book
        self.last_book = book
        
        # Store mid price
        self.mid_prices.append((book.timestamp, book.mid_price))
        
        # Cleanup old data
        self._cleanup_old_data(book.timestamp)
        
        # Compute returns
        self._compute_returns()
        
        # Check if we should emit feature
        if not self._should_emit_feature(book.timestamp):
            return None
        
        # Compute features
        row = self._compute_features(book)
        self.last_feature_time = book.timestamp
        return row
    
    def _on_trade(self, trade: TradeEvent) -> Optional[FeatureRow]:
        """Process trade event."""
        if trade.symbol != self.symbol:
            return None
        
        self.trades.append(trade)
        return None  # Trades don't directly emit features
    
    def _compute_features(self, book: L2BookSnapshotEvent) -> FeatureRow:
        """Compute all features for current state."""
        # Recent trades for flow features
        recent_trades = self._get_recent_trades(book.timestamp, self.window_seconds)
        
        # Price features
        mid = ms.compute_mid_price(book)
        micro = ms.compute_microprice(book)
        spread = ms.compute_spread(book)
        spread_bps = ms.compute_spread_bps(book)
        
        # Book features
        imbalance = ms.compute_book_imbalance(book)
        depth_imb = ms.compute_depth_imbalance(book, levels=5)
        
        # OFI - need trades since last book
        trades_since_book = []
        if self.prev_book is not None:
            trades_since_book = [
                t for t in self.trades
                if self.prev_book.timestamp <= t.timestamp <= book.timestamp
            ]
        ofi = ms.compute_ofi(self.prev_book, book, trades_since_book)
        
        # Trade features
        trade_imb = ms.compute_trade_imbalance(recent_trades)
        trade_intensity = ms.compute_trade_intensity(recent_trades)
        
        # Volatility features
        vol = ms.compute_realized_volatility(self.returns, window=20)
        
        # Returns (1-second equivalent)
        returns_1s = self.returns[-1] if self.returns else 0.0
        
        # Z-score
        zscore = ms.compute_rolling_zscore([p for _, p in self.mid_prices], window=20)
        
        return FeatureRow(
            timestamp=book.timestamp,
            symbol=book.symbol,
            mid_price=mid,
            microprice=micro,
            spread=spread,
            spread_bps=spread_bps,
            book_imbalance=imbalance,
            depth_imbalance_5=depth_imb,
            ofi=ofi,
            trade_imbalance=trade_imb,
            trade_intensity=trade_intensity,
            realized_vol_20=vol,
            returns_1s=returns_1s,
            zscore_mid_20=zscore,
        )
    
    def process_events(
        self, events: List[Event]
    ) -> pd.DataFrame:
        """Process a list of events and return feature DataFrame.
        
        Args:
            events: List of events (books and trades).
        
        Returns:
            DataFrame with features indexed by timestamp.
        """
        rows = []
        for event in events:
            row = self.on_event(event)
            if row is not None:
                rows.append(row.to_dict())
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        df.set_index("timestamp", inplace=True)
        return df
