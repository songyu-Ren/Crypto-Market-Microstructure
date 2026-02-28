"""Data loader interface for CSV and Parquet historical data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Iterator, List
import numpy as np
import pandas as pd

from crypto_mm_research.data.events import L2BookSnapshotEvent, TradeEvent, Side


class DataLoader(ABC):
    """Abstract base class for market data loaders."""
    
    @abstractmethod
    def load_books(self, symbol: str, start: datetime, end: datetime) -> Iterator[L2BookSnapshotEvent]:
        """Load L2 book snapshot events."""
        pass
    
    @abstractmethod
    def load_trades(self, symbol: str, start: datetime, end: datetime) -> Iterator[TradeEvent]:
        """Load trade events."""
        pass
    
    @abstractmethod
    def load_merged(
        self, symbol: str, start: datetime, end: datetime
    ) -> Iterator[L2BookSnapshotEvent | TradeEvent]:
        """Load and merge books and trades, sorted by timestamp."""
        pass


class CSVDataLoader(DataLoader):
    """Load market data from CSV files.
    
    Expected CSV formats:
    
    Books CSV:
    - timestamp (ISO format)
    - symbol
    - bid_price_0, bid_size_0, bid_price_1, bid_size_1, ...
    - ask_price_0, ask_size_0, ask_price_1, ask_size_1, ...
    
    Trades CSV:
    - timestamp (ISO format)
    - symbol
    - price
    - size
    - side ("buy" or "sell")
    - trade_id (optional)
    """
    
    def __init__(
        self,
        books_path: str | Path | None = None,
        trades_path: str | Path | None = None,
        book_levels: int = 10,
    ) -> None:
        """Initialize CSV loader.
        
        Args:
            books_path: Path to books CSV file.
            trades_path: Path to trades CSV file.
            book_levels: Expected number of book levels.
        """
        self.books_path = Path(books_path) if books_path else None
        self.trades_path = Path(trades_path) if trades_path else None
        self.book_levels = book_levels
    
    def _parse_books_df(self, df: pd.DataFrame) -> Iterator[L2BookSnapshotEvent]:
        """Parse books DataFrame into events."""
        for _, row in df.iterrows():
            timestamp = pd.to_datetime(row["timestamp"]).to_pydatetime()
            symbol = str(row["symbol"])
            
            # Extract bid levels
            bids = []
            for i in range(self.book_levels):
                price = row.get(f"bid_price_{i}", 0.0)
                size = row.get(f"bid_size_{i}", 0.0)
                if price > 0 and size > 0:
                    bids.append([float(price), float(size)])
            
            # Extract ask levels
            asks = []
            for i in range(self.book_levels):
                price = row.get(f"ask_price_{i}", 0.0)
                size = row.get(f"ask_size_{i}", 0.0)
                if price > 0 and size > 0:
                    asks.append([float(price), float(size)])
            
            yield L2BookSnapshotEvent(
                timestamp=timestamp,
                symbol=symbol,
                bids=np.array(bids) if bids else np.array([]).reshape(0, 2),
                asks=np.array(asks) if asks else np.array([]).reshape(0, 2),
            )
    
    def _parse_trades_df(self, df: pd.DataFrame) -> Iterator[TradeEvent]:
        """Parse trades DataFrame into events."""
        for _, row in df.iterrows():
            timestamp = pd.to_datetime(row["timestamp"]).to_pydatetime()
            symbol = str(row["symbol"])
            price = float(row["price"])
            size = float(row["size"])
            side_str = str(row["side"]).lower()
            side = Side.BUY if side_str in ("buy", "b", "1") else Side.SELL
            trade_id = str(row.get("trade_id", ""))
            
            yield TradeEvent(
                timestamp=timestamp,
                symbol=symbol,
                price=price,
                size=size,
                side=side,
                trade_id=trade_id,
            )
    
    def load_books(self, symbol: str, start: datetime, end: datetime) -> Iterator[L2BookSnapshotEvent]:
        """Load L2 book snapshot events from CSV."""
        if self.books_path is None:
            return iter([])
        
        df = pd.read_csv(self.books_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
        df = df[df["symbol"] == symbol]
        df = df.sort_values("timestamp")
        
        yield from self._parse_books_df(df)
    
    def load_trades(self, symbol: str, start: datetime, end: datetime) -> Iterator[TradeEvent]:
        """Load trade events from CSV."""
        if self.trades_path is None:
            return iter([])
        
        df = pd.read_csv(self.trades_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
        df = df[df["symbol"] == symbol]
        df = df.sort_values("timestamp")
        
        yield from self._parse_trades_df(df)
    
    def load_merged(
        self, symbol: str, start: datetime, end: datetime
    ) -> Iterator[L2BookSnapshotEvent | TradeEvent]:
        """Load and merge books and trades, sorted by timestamp."""
        books = list(self.load_books(symbol, start, end))
        trades = list(self.load_trades(symbol, start, end))
        
        # Merge sort by timestamp
        all_events = books + trades
        all_events.sort(key=lambda x: x.timestamp)
        
        yield from all_events


class ParquetDataLoader(DataLoader):
    """Load market data from Parquet files.
    
    Same schema as CSV but stored in Parquet format for efficiency.
    """
    
    def __init__(
        self,
        books_path: str | Path | None = None,
        trades_path: str | Path | None = None,
        book_levels: int = 10,
    ) -> None:
        """Initialize Parquet loader."""
        self.books_path = Path(books_path) if books_path else None
        self.trades_path = Path(trades_path) if trades_path else None
        self.book_levels = book_levels
    
    def _parse_books_df(self, df: pd.DataFrame) -> Iterator[L2BookSnapshotEvent]:
        """Parse books DataFrame into events."""
        for _, row in df.iterrows():
            timestamp = row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"]
            symbol = str(row["symbol"])
            
            bids = []
            asks = []
            for i in range(self.book_levels):
                bid_price = row.get(f"bid_price_{i}", 0.0)
                bid_size = row.get(f"bid_size_{i}", 0.0)
                ask_price = row.get(f"ask_price_{i}", 0.0)
                ask_size = row.get(f"ask_size_{i}", 0.0)
                
                if bid_price > 0 and bid_size > 0:
                    bids.append([float(bid_price), float(bid_size)])
                if ask_price > 0 and ask_size > 0:
                    asks.append([float(ask_price), float(ask_size)])
            
            yield L2BookSnapshotEvent(
                timestamp=timestamp,
                symbol=symbol,
                bids=np.array(bids) if bids else np.array([]).reshape(0, 2),
                asks=np.array(asks) if asks else np.array([]).reshape(0, 2),
            )
    
    def _parse_trades_df(self, df: pd.DataFrame) -> Iterator[TradeEvent]:
        """Parse trades DataFrame into events."""
        for _, row in df.iterrows():
            timestamp = row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"]
            symbol = str(row["symbol"])
            price = float(row["price"])
            size = float(row["size"])
            side_str = str(row["side"]).lower()
            side = Side.BUY if side_str in ("buy", "b", "1") else Side.SELL
            trade_id = str(row.get("trade_id", ""))
            
            yield TradeEvent(
                timestamp=timestamp,
                symbol=symbol,
                price=price,
                size=size,
                side=side,
                trade_id=trade_id,
            )
    
    def load_books(self, symbol: str, start: datetime, end: datetime) -> Iterator[L2BookSnapshotEvent]:
        """Load L2 book snapshot events from Parquet."""
        if self.books_path is None:
            return iter([])
        
        df = pd.read_parquet(self.books_path)
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
        df = df[df["symbol"] == symbol]
        df = df.sort_values("timestamp")
        
        yield from self._parse_books_df(df)
    
    def load_trades(self, symbol: str, start: datetime, end: datetime) -> Iterator[TradeEvent]:
        """Load trade events from Parquet."""
        if self.trades_path is None:
            return iter([])
        
        df = pd.read_parquet(self.trades_path)
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
        df = df[df["symbol"] == symbol]
        df = df.sort_values("timestamp")
        
        yield from self._parse_trades_df(df)
    
    def load_merged(
        self, symbol: str, start: datetime, end: datetime
    ) -> Iterator[L2BookSnapshotEvent | TradeEvent]:
        """Load and merge books and trades, sorted by timestamp."""
        books = list(self.load_books(symbol, start, end))
        trades = list(self.load_trades(symbol, start, end))
        
        all_events = books + trades
        all_events.sort(key=lambda x: x.timestamp)
        
        yield from all_events
