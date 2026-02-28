"""CSV stub adapter for example data format."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterator, Dict, Any
import pandas as pd
import numpy as np

from crypto_mm_research.adapters.base import BaseMarketDataAdapter
from crypto_mm_research.data.events import L2BookSnapshotEvent, TradeEvent, Side


class CSVStubAdapter(BaseMarketDataAdapter):
    """Adapter for CSV stub format."""
    
    def __init__(
        self,
        book_levels: int = 10,
        symbol_mapping: Dict[str, str] | None = None,
    ) -> None:
        super().__init__(symbol_mapping)
        self.book_levels = book_levels
    
    def get_required_columns(self) -> Dict[str, str]:
        return {
            "event_ts": "Event timestamp (ISO format)",
            "type": "Event type: 'book' or 'trade'",
            "symbol": "Trading pair symbol",
            "bid_price_0": "Best bid price (for book events)",
            "bid_size_0": "Best bid size (for book events)",
            "ask_price_0": "Best ask price (for book events)",
            "ask_size_0": "Best ask size (for book events)",
            "price": "Trade price (for trade events)",
            "size": "Trade size (for trade events)",
            "side": "Trade side: 'buy' or 'sell' (for trade events)",
        }
    
    def load_books(
        self,
        filepath: str | Path,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Iterator[L2BookSnapshotEvent]:
        df = pd.read_csv(filepath)
        df["event_ts"] = pd.to_datetime(df["event_ts"])
        df = df[df["type"] == "book"]
        
        if start:
            df = df[df["event_ts"] >= start]
        if end:
            df = df[df["event_ts"] <= end]
        
        df = df.sort_values("event_ts")
        
        for _, row in df.iterrows():
            yield self._parse_book_row(row)
    
    def load_trades(
        self,
        filepath: str | Path,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Iterator[TradeEvent]:
        df = pd.read_csv(filepath)
        df["event_ts"] = pd.to_datetime(df["event_ts"])
        df = df[df["type"] == "trade"]
        
        if start:
            df = df[df["event_ts"] >= start]
        if end:
            df = df[df["event_ts"] <= end]
        
        df = df.sort_values("event_ts")
        
        for _, row in df.iterrows():
            yield self._parse_trade_row(row)
    
    def load_merged(
        self,
        filepath: str | Path,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Iterator[L2BookSnapshotEvent | TradeEvent]:
        books = list(self.load_books(filepath, start, end))
        trades = list(self.load_trades(filepath, start, end))
        
        all_events = books + trades
        all_events.sort(key=lambda x: x.timestamp)
        
        yield from all_events
    
    def _parse_book_row(self, row: pd.Series) -> L2BookSnapshotEvent:
        timestamp = row["event_ts"].to_pydatetime() if hasattr(row["event_ts"], "to_pydatetime") else row["event_ts"]
        symbol = self.normalize_symbol(row["symbol"])
        
        bids = []
        asks = []
        
        for i in range(self.book_levels):
            bid_price = row.get(f"bid_price_{i}", 0)
            bid_size = row.get(f"bid_size_{i}", 0)
            ask_price = row.get(f"ask_price_{i}", 0)
            ask_size = row.get(f"ask_size_{i}", 0)
            
            if bid_price > 0 and bid_size > 0:
                bids.append([float(bid_price), float(bid_size)])
            if ask_price > 0 and ask_size > 0:
                asks.append([float(ask_price), float(ask_size)])
        
        return L2BookSnapshotEvent(
            timestamp=timestamp,
            symbol=symbol,
            bids=np.array(bids) if bids else np.array([]).reshape(0, 2),
            asks=np.array(asks) if asks else np.array([]).reshape(0, 2),
        )
    
    def _parse_trade_row(self, row: pd.Series) -> TradeEvent:
        timestamp = row["event_ts"].to_pydatetime() if hasattr(row["event_ts"], "to_pydatetime") else row["event_ts"]
        symbol = self.normalize_symbol(row["symbol"])
        side = Side.BUY if row["side"] == "buy" else Side.SELL
        
        return TradeEvent(
            timestamp=timestamp,
            symbol=symbol,
            price=float(row["price"]),
            size=float(row["size"]),
            side=side,
        )
