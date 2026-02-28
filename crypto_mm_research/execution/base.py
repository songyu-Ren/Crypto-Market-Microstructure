"""Base execution model and order state definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any
import numpy as np


class OrderStatus(Enum):
    """Order lifecycle states."""
    PENDING_SUBMIT = auto()
    ACTIVE = auto()
    PARTIALLY_FILLED = auto()
    PENDING_CANCEL = auto()
    PENDING_REPLACE = auto()
    CANCELED = auto()
    FILLED = auto()
    REJECTED = auto()


class TimeInForce(Enum):
    """Time in force options."""
    GTC = "good_till_cancel"
    IOC = "immediate_or_cancel"
    FOK = "fill_or_kill"


@dataclass
class Order:
    """Order with full lifecycle tracking."""
    
    client_order_id: str
    symbol: str
    side: str
    price: float
    size: float
    time_in_force: TimeInForce = TimeInForce.GTC
    
    status: OrderStatus = OrderStatus.PENDING_SUBMIT
    exchange_order_id: Optional[str] = None
    
    submit_ts: Optional[datetime] = None
    active_ts: Optional[datetime] = None
    cancel_ts: Optional[datetime] = None
    cancel_effective_ts: Optional[datetime] = None
    last_fill_ts: Optional[datetime] = None
    
    filled_size: float = 0.0
    remaining_size: float = field(init=False)
    avg_fill_price: float = 0.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.remaining_size = self.size - self.filled_size
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in (
            OrderStatus.ACTIVE,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.PENDING_SUBMIT,
        )
    
    @property
    def is_done(self) -> bool:
        """Check if order is complete."""
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
        )
    
    def update_fill(self, fill_size: float, fill_price: float, fill_ts: datetime) -> None:
        """Update order with new fill."""
        self.filled_size += fill_size
        self.remaining_size = self.size - self.filled_size
        
        total_value = self.avg_fill_price * (self.filled_size - fill_size) + fill_price * fill_size
        self.avg_fill_price = total_value / self.filled_size if self.filled_size > 0 else 0
        
        self.last_fill_ts = fill_ts
        
        if self.remaining_size <= 1e-9:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED


@dataclass(frozen=True)
class FillEvent:
    """A fill event with detailed metadata."""
    
    fill_id: str
    order_id: str
    client_order_id: str
    symbol: str
    
    fill_ts: datetime
    arrival_ts: datetime
    
    price: float
    size: float
    side: str
    
    is_maker: bool
    fee: float
    
    mid_at_fill: float
    spread_at_fill: float
    book_state_hash: Optional[str] = None
    
    queue_ahead_before: Optional[float] = None
    queue_ahead_after: Optional[float] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExecutionModel(ABC):
    """Abstract base class for execution models."""
    
    def __init__(
        self,
        tick_size: float = 0.1,
        lot_size: float = 0.001,
        maker_fee_rate: float = 0.0002,
        taker_fee_rate: float = 0.0005,
    ) -> None:
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.maker_fee_rate = maker_fee_rate
        self.taker_fee_rate = taker_fee_rate
        self._fill_counter = 0
    
    def _round_price(self, price: float) -> float:
        return round(price / self.tick_size) * self.tick_size
    
    def _round_size(self, size: float) -> float:
        return max(self.lot_size, round(size / self.lot_size) * self.lot_size)
    
    def _generate_fill_id(self) -> str:
        self._fill_counter += 1
        return f"F{self._fill_counter:010d}"
    
    @abstractmethod
    def submit_order(self, order: Order, event_ts: datetime) -> List[FillEvent]:
        pass
    
    @abstractmethod
    def cancel_order(self, order: Order, event_ts: datetime) -> bool:
        pass
    
    @abstractmethod
    def replace_order(
        self,
        old_order: Order,
        new_price: float,
        new_size: float,
        event_ts: datetime,
    ) -> Optional[Order]:
        pass
    
    @abstractmethod
    def on_book_update(
        self,
        orders: Dict[str, Order],
        book,
        event_ts: datetime,
    ) -> List[FillEvent]:
        pass
    
    @abstractmethod
    def on_trade(
        self,
        orders: Dict[str, Order],
        trade,
        event_ts: datetime,
    ) -> List[FillEvent]:
        pass
    
    def compute_fee(self, notional: float, is_maker: bool) -> float:
        rate = self.maker_fee_rate if is_maker else self.taker_fee_rate
        return notional * rate
