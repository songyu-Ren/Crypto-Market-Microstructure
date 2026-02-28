"""Event dataclasses for market microstructure data."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Union
import numpy as np


class Side(Enum):
    """Trade side: buyer-initiated or seller-initiated."""
    
    BUY = 1
    SELL = -1


@dataclass(frozen=True)
class L2BookSnapshotEvent:
    """L2 order book snapshot event.
    
    Attributes:
        timestamp: Event timestamp (timezone-aware UTC recommended).
        symbol: Trading pair symbol (e.g., "BTC-USDT").
        bids: Array of [price, size] for bid levels, sorted by price descending.
        asks: Array of [price, size] for ask levels, sorted by price ascending.
        sequence: Optional sequence number for ordering.
    """
    
    timestamp: datetime
    symbol: str
    bids: np.ndarray  # Shape: (n_levels, 2) - [price, size]
    asks: np.ndarray  # Shape: (n_levels, 2) - [price, size]
    sequence: int = 0
    
    def __post_init__(self) -> None:
        """Validate book structure."""
        if self.bids.size > 0 and self.asks.size > 0:
            best_bid = self.bids[0, 0]
            best_ask = self.asks[0, 0]
            if best_bid >= best_ask:
                raise ValueError(
                    f"Best bid ({best_bid}) must be less than best ask ({best_ask})"
                )
    
    @property
    def best_bid(self) -> float:
        """Return best bid price."""
        return float(self.bids[0, 0]) if self.bids.size > 0 else 0.0
    
    @property
    def best_ask(self) -> float:
        """Return best ask price."""
        return float(self.asks[0, 0]) if self.asks.size > 0 else 0.0
    
    @property
    def mid_price(self) -> float:
        """Return mid price."""
        if self.bids.size == 0 or self.asks.size == 0:
            return 0.0
        return (self.best_bid + self.best_ask) / 2.0
    
    @property
    def spread(self) -> float:
        """Return bid-ask spread."""
        if self.bids.size == 0 or self.asks.size == 0:
            return 0.0
        return self.best_ask - self.best_bid
    
    @property
    def microprice(self) -> float:
        """Return volume-weighted microprice.
        
        Microprice is a weighted average of best bid and ask,
        weighted by inverse volume (more weight to side with less volume).
        """
        if self.bids.size == 0 or self.asks.size == 0:
            return self.mid_price
        
        bid_size = self.bids[0, 1]
        ask_size = self.asks[0, 1]
        
        if bid_size + ask_size == 0:
            return self.mid_price
        
        # Weight by inverse size (more aggressive side has less size)
        bid_weight = ask_size / (bid_size + ask_size)
        ask_weight = bid_size / (bid_size + ask_size)
        
        return bid_weight * self.best_bid + ask_weight * self.best_ask


@dataclass(frozen=True)
class TradeEvent:
    """Trade event representing a market execution.
    
    Attributes:
        timestamp: Event timestamp.
        symbol: Trading pair symbol.
        price: Trade price.
        size: Trade size (always positive).
        side: Trade side (BUY for buyer-initiated, SELL for seller-initiated).
        trade_id: Optional trade identifier.
    """
    
    timestamp: datetime
    symbol: str
    price: float
    size: float
    side: Side
    trade_id: str = ""
    
    def __post_init__(self) -> None:
        """Validate trade data."""
        if self.size <= 0:
            raise ValueError(f"Trade size must be positive, got {self.size}")
        if self.price <= 0:
            raise ValueError(f"Trade price must be positive, got {self.price}")


# Union type for all events
Event = Union[L2BookSnapshotEvent, TradeEvent]
