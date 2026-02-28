"""Strategy interface and implementations for market making."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from crypto_mm_research.data.events import L2BookSnapshotEvent, TradeEvent
from crypto_mm_research.backtest.account import Account


@dataclass
class Quote:
    """A single quote (order) to place."""
    
    price: float
    size: float
    
    def is_valid(self) -> bool:
        """Check if quote is valid."""
        return self.price > 0 and self.size > 0


@dataclass
class StrategyOutput:
    """Output from strategy on each book update."""
    
    bid: Optional[Quote] = None
    ask: Optional[Quote] = None
    
    def has_quotes(self) -> bool:
        """Check if any quotes are present."""
        return (self.bid is not None and self.bid.is_valid()) or \
               (self.ask is not None and self.ask.is_valid())


class Strategy(ABC):
    """Abstract base class for trading strategies."""
    
    @abstractmethod
    def on_book(
        self,
        timestamp: datetime,
        book: L2BookSnapshotEvent,
        account: Account,
    ) -> StrategyOutput:
        """Process book update and return desired quotes.
        
        Args:
            timestamp: Current timestamp.
            book: Current L2 book snapshot.
            account: Current account state.
        
        Returns:
            StrategyOutput with desired bid/ask quotes.
        """
        pass
    
    def on_trade(
        self,
        timestamp: datetime,
        trade: TradeEvent,
        account: Account,
    ) -> None:
        """Optional: process trade events.
        
        Args:
            timestamp: Current timestamp.
            trade: Trade event.
            account: Current account state.
        """
        pass
    
    def reset(self) -> None:
        """Reset strategy state for new backtest."""
        pass


class MarketMakingStrategy(Strategy):
    """Baseline market making strategy with inventory skew.
    
    This strategy:
    1. Quotes bid/ask around mid price
    2. Adjusts quotes based on inventory (skew)
    3. Optionally widens spread during high volatility
    4. Respects inventory limits
    """
    
    def __init__(
        self,
        target_half_spread_bps: float = 5.0,
        quote_size: float = 0.1,
        skew_coeff: float = 1.0,
        inventory_limit: float = 1.0,
        vol_adaptive: bool = False,
        vol_threshold: float = 0.001,
        min_half_spread_bps: float = 2.0,
    ) -> None:
        """Initialize market making strategy.
        
        Args:
            target_half_spread_bps: Target half-spread in basis points.
            quote_size: Size to quote on each side.
            skew_coeff: Inventory skew coefficient (0 = no skew).
            inventory_limit: Maximum absolute inventory before stopping quotes.
            vol_adaptive: Whether to adapt spread to volatility.
            vol_threshold: Volatility threshold for spread widening.
            min_half_spread_bps: Minimum half-spread in bps.
        """
        self.target_half_spread_bps = target_half_spread_bps / 10000.0
        self.quote_size = quote_size
        self.skew_coeff = skew_coeff
        self.inventory_limit = inventory_limit
        self.vol_adaptive = vol_adaptive
        self.vol_threshold = vol_threshold
        self.min_half_spread_bps = min_half_spread_bps / 10000.0
        
        # State
        self.current_vol: float = 0.0
        self.last_mid: float = 0.0
        self.returns: list[float] = []
    
    def _update_volatility(self, mid: float) -> None:
        """Update volatility estimate."""
        if self.last_mid > 0:
            ret = (mid - self.last_mid) / self.last_mid
            self.returns.append(abs(ret))
            if len(self.returns) > 20:
                self.returns.pop(0)
        
        self.current_vol = sum(self.returns) / len(self.returns) if self.returns else 0.0
        self.last_mid = mid
    
    def _compute_half_spread(self, mid: float) -> float:
        """Compute half-spread in price terms."""
        base_spread = mid * self.target_half_spread_bps
        
        if self.vol_adaptive and self.current_vol > self.vol_threshold:
            # Widen spread during high volatility
            vol_multiplier = 1.0 + (self.current_vol / self.vol_threshold - 1.0)
            base_spread *= vol_multiplier
        
        min_spread = mid * self.min_half_spread_bps
        return max(base_spread, min_spread)
    
    def _compute_skew_offset(self, inventory: float, mid: float) -> float:
        """Compute inventory skew offset.
        
        Positive inventory (long) -> shift quotes down (more willing to sell)
        Negative inventory (short) -> shift quotes up (more willing to buy)
        """
        if abs(inventory) < 1e-9:
            return 0.0
        
        # Skew is proportional to inventory
        # Capped at half spread to avoid crossing
        half_spread = self._compute_half_spread(mid)
        max_skew = half_spread * 0.8
        
        raw_skew = -inventory * self.skew_coeff * mid * 0.001  # Scale factor
        return max(-max_skew, min(max_skew, raw_skew))
    
    def on_book(
        self,
        timestamp: datetime,
        book: L2BookSnapshotEvent,
        account: Account,
    ) -> StrategyOutput:
        """Generate quotes based on current book and inventory."""
        mid = book.mid_price
        if mid <= 0:
            return StrategyOutput()
        
        # Update volatility
        self._update_volatility(mid)
        
        # Get current inventory
        position = account.get_position(book.symbol)
        inventory = position.size
        
        # Check inventory limit
        if abs(inventory) >= self.inventory_limit:
            # Stop quoting on the side that would increase exposure
            output = StrategyOutput()
            
            half_spread = self._compute_half_spread(mid)
            skew = self._compute_skew_offset(inventory, mid)
            
            if inventory > 0:
                # Long, only quote ask
                ask_price = mid + half_spread + skew
                output.ask = Quote(price=ask_price, size=self.quote_size)
            else:
                # Short, only quote bid
                bid_price = mid - half_spread + skew
                output.bid = Quote(price=bid_price, size=self.quote_size)
            
            return output
        
        # Normal quoting
        half_spread = self._compute_half_spread(mid)
        skew = self._compute_skew_offset(inventory, mid)
        
        bid_price = mid - half_spread + skew
        ask_price = mid + half_spread + skew
        
        # Ensure we don't cross the spread
        bid_price = min(bid_price, book.best_bid)
        ask_price = max(ask_price, book.best_ask)
        
        return StrategyOutput(
            bid=Quote(price=bid_price, size=self.quote_size),
            ask=Quote(price=ask_price, size=self.quote_size),
        )
    
    def reset(self) -> None:
        """Reset strategy state."""
        self.current_vol = 0.0
        self.last_mid = 0.0
        self.returns = []
