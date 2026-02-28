"""Base adapter for market data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Iterator, Dict, Any

from crypto_mm_research.data.events import L2BookSnapshotEvent, TradeEvent, Event


class BaseMarketDataAdapter(ABC):
    """Base class for market data adapters."""
    
    def __init__(self, symbol_mapping: Dict[str, str] | None = None) -> None:
        self.symbol_mapping = symbol_mapping or {}
    
    def normalize_symbol(self, exchange_symbol: str) -> str:
        return self.symbol_mapping.get(exchange_symbol, exchange_symbol)
    
    @abstractmethod
    def load_books(
        self,
        filepath: str | Path,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Iterator[L2BookSnapshotEvent]:
        pass
    
    @abstractmethod
    def load_trades(
        self,
        filepath: str | Path,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Iterator[TradeEvent]:
        pass
    
    @abstractmethod
    def load_merged(
        self,
        filepath: str | Path,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Iterator[Event]:
        pass
    
    @abstractmethod
    def get_required_columns(self) -> Dict[str, str]:
        pass
