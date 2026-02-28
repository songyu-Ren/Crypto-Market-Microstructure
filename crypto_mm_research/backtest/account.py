"""Account and fill tracking for backtest."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict
from enum import Enum


class Side(Enum):
    """Position side."""
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass(frozen=True)
class Fill:
    """A single fill/execution."""
    
    timestamp: datetime
    symbol: str
    price: float
    size: float  # Always positive
    side: Side  # From strategy perspective
    fee: float
    is_maker: bool
    
    @property
    def notional(self) -> float:
        """Notional value of fill."""
        return self.price * self.size
    
    @property
    def pnl(self) -> float:
        """PnL contribution (excluding fees)."""
        # This is computed by account
        return 0.0


@dataclass
class Position:
    """Track position state."""
    
    symbol: str
    size: float = 0.0  # Positive = long, negative = short
    avg_entry_price: float = 0.0
    realized_pnl: float = 0.0
    
    @property
    def side(self) -> Side:
        """Current position side."""
        if self.size > 1e-9:
            return Side.LONG
        elif self.size < -1e-9:
            return Side.SHORT
        return Side.FLAT
    
    @property
    def notional(self) -> float:
        """Position notional value."""
        return abs(self.size) * self.avg_entry_price if self.size != 0 else 0.0
    
    def update(self, fill: Fill) -> None:
        """Update position with new fill."""
        fill_size_signed = fill.size * fill.side.value
        
        if abs(self.size) < 1e-9:
            # Opening position
            self.size = fill_size_signed
            self.avg_entry_price = fill.price
        elif (self.size > 0 and fill_size_signed > 0) or (self.size < 0 and fill_size_signed < 0):
            # Adding to position
            old_notional = abs(self.size) * self.avg_entry_price
            new_notional = fill.size * fill.price
            total_size = abs(self.size) + fill.size
            self.avg_entry_price = (old_notional + new_notional) / total_size
            self.size += fill_size_signed
        else:
            # Reducing or flipping position
            close_size = min(abs(self.size), fill.size)
            
            # Realized PnL on closed portion
            if self.size > 0:
                # Long position, selling
                self.realized_pnl += close_size * (fill.price - self.avg_entry_price)
            else:
                # Short position, buying
                self.realized_pnl += close_size * (self.avg_entry_price - fill.price)
            
            # Subtract fees
            self.realized_pnl -= fill.fee
            
            remaining = fill.size - close_size
            
            if remaining < 1e-9:
                # Fully closed
                self.size = 0.0
                self.avg_entry_price = 0.0
            else:
                # Flipped position
                self.size = fill_size_signed - (self.size)  # New signed size
                self.avg_entry_price = fill.price


@dataclass
class Account:
    """Trading account with cash, positions, and PnL tracking."""
    
    initial_cash: float = 100000.0
    maker_fee_rate: float = 0.0002  # 2 bps maker fee
    taker_fee_rate: float = 0.0005  # 5 bps taker fee
    
    cash: float = field(init=False)
    positions: Dict[str, Position] = field(default_factory=dict)
    fills: List[Fill] = field(default_factory=list)
    equity_curve: List[Dict] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Initialize account."""
        self.cash = self.initial_cash
    
    def get_position(self, symbol: str) -> Position:
        """Get or create position for symbol."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]
    
    @property
    def total_position(self) -> float:
        """Total position size across all symbols."""
        return sum(pos.size for pos in self.positions.values())
    
    @property
    def gross_notional(self) -> float:
        """Gross notional exposure."""
        return sum(pos.notional for pos in self.positions.values())
    
    def apply_fill(
        self,
        timestamp: datetime,
        symbol: str,
        price: float,
        size: float,
        side: Side,
        is_maker: bool = True,
    ) -> Fill:
        """Apply a fill to the account.
        
        Args:
            timestamp: Fill timestamp.
            symbol: Trading symbol.
            price: Fill price.
            size: Fill size (always positive).
            side: Side from strategy perspective (BUY=long, SELL=short).
            is_maker: Whether this is a maker fill.
        
        Returns:
            The Fill object.
        """
        notional = price * size
        fee_rate = self.maker_fee_rate if is_maker else self.taker_fee_rate
        fee = notional * fee_rate
        
        fill = Fill(
            timestamp=timestamp,
            symbol=symbol,
            price=price,
            size=size,
            side=side,
            fee=fee,
            is_maker=is_maker,
        )
        
        # Update position
        position = self.get_position(symbol)
        old_realized = position.realized_pnl
        position.update(fill)
        
        # Update cash
        if side == Side.LONG:
            # Buying: pay notional + fee
            self.cash -= notional + fee
        else:
            # Selling: receive notional - fee
            self.cash += notional - fee
        
        self.fills.append(fill)
        return fill
    
    def mark_to_market(self, timestamp: datetime, prices: Dict[str, float]) -> None:
        """Mark positions to market and record equity.
        
        Args:
            timestamp: Current timestamp.
            prices: Dict of symbol -> mark price.
        """
        unrealized_pnl = 0.0
        
        for symbol, position in self.positions.items():
            if abs(position.size) < 1e-9:
                continue
            
            mark_price = prices.get(symbol, position.avg_entry_price)
            
            if position.size > 0:
                # Long: unrealized = size * (mark - entry)
                unrealized_pnl += position.size * (mark_price - position.avg_entry_price)
            else:
                # Short: unrealized = size * (entry - mark)
                unrealized_pnl += abs(position.size) * (position.avg_entry_price - mark_price)
        
        total_equity = self.cash + unrealized_pnl
        
        for symbol, price in prices.items():
            position = self.positions.get(symbol)
            position_size = position.size if position else 0.0
            
            self.equity_curve.append({
                "timestamp": timestamp,
                "symbol": symbol,
                "cash": self.cash,
                "position": position_size,
                "mark_price": price,
                "unrealized_pnl": unrealized_pnl,
                "realized_pnl": sum(p.realized_pnl for p in self.positions.values()),
                "total_equity": total_equity,
            })
    
    def get_stats(self) -> Dict:
        """Get account statistics."""
        total_realized = sum(p.realized_pnl for p in self.positions.values())
        
        return {
            "cash": self.cash,
            "total_realized_pnl": total_realized,
            "n_fills": len(self.fills),
            "n_maker_fills": sum(1 for f in self.fills if f.is_maker),
            "total_fees": sum(f.fee for f in self.fills),
        }
