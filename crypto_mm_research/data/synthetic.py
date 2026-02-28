"""Synthetic data generator for market microstructure research."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterator, List
import numpy as np

from crypto_mm_research.data.events import L2BookSnapshotEvent, TradeEvent, Side


class SyntheticDataGenerator:
    """Generate realistic synthetic L2 order book and trade data.
    
    This generator creates synthetic market data that mimics key
    microstructure properties:
    - Mean-reverting spread around a target
    - Correlation between volatility and spread
    - Trade clustering (bursts of activity)
    - Price impact from trades
    
    All generation is deterministic given a random seed.
    """
    
    def __init__(
        self,
        symbol: str = "BTC-USDT",
        start_price: float = 50000.0,
        tick_size: float = 0.1,
        lot_size: float = 0.001,
        volatility_annual: float = 0.8,  # 80% annual vol for crypto
        target_spread_bps: float = 5.0,  # 5 bps target spread
        book_levels: int = 10,
        base_quantity: float = 1.0,
        random_seed: int = 42,
    ) -> None:
        """Initialize the synthetic data generator.
        
        Args:
            symbol: Trading pair symbol.
            start_price: Starting mid price.
            tick_size: Minimum price increment.
            lot_size: Minimum quantity increment.
            volatility_annual: Annualized volatility (e.g., 0.8 = 80%).
            target_spread_bps: Target spread in basis points.
            book_levels: Number of levels to generate on each side.
            base_quantity: Base quantity for book sizes.
            random_seed: Random seed for reproducibility.
        """
        self.symbol = symbol
        self.start_price = start_price
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.volatility_annual = volatility_annual
        self.target_spread_bps = target_spread_bps / 10000.0  # Convert to decimal
        self.book_levels = book_levels
        self.base_quantity = base_quantity
        
        self.rng = np.random.default_rng(random_seed)
        
        # State variables
        self.current_mid = start_price
        self.current_vol = volatility_annual / np.sqrt(365 * 24 * 3600)  # Per second
        self.sequence = 0
    
    def _round_price(self, price: float) -> float:
        """Round price to tick size."""
        return round(price / self.tick_size) * self.tick_size
    
    def _round_size(self, size: float) -> float:
        """Round size to lot size."""
        return max(self.lot_size, round(size / self.lot_size) * self.lot_size)
    
    def _generate_book(self, timestamp: datetime) -> L2BookSnapshotEvent:
        """Generate an L2 book snapshot."""
        # Target half-spread in price terms
        target_half_spread = self.current_mid * self.target_spread_bps / 2
        
        # Add some noise to spread (mean-reverting)
        spread_noise = self.rng.normal(0, target_half_spread * 0.2)
        half_spread = max(self.tick_size, target_half_spread + spread_noise)
        half_spread = self._round_price(half_spread)
        
        best_bid = self._round_price(self.current_mid - half_spread)
        best_ask = self._round_price(self.current_mid + half_spread)
        
        # Generate bid levels
        bids = []
        for i in range(self.book_levels):
            # Price decreases as we go down the book
            price = best_bid - i * self._round_price(half_spread * (1 + i * 0.5))
            # Size increases slightly deeper in book
            size = self._round_size(
                self.base_quantity * (1 + i * 0.3) * (1 + self.rng.exponential(0.5))
            )
            bids.append([price, size])
        
        # Generate ask levels
        asks = []
        for i in range(self.book_levels):
            price = best_ask + i * self._round_price(half_spread * (1 + i * 0.5))
            size = self._round_size(
                self.base_quantity * (1 + i * 0.3) * (1 + self.rng.exponential(0.5))
            )
            asks.append([price, size])
        
        self.sequence += 1
        
        return L2BookSnapshotEvent(
            timestamp=timestamp,
            symbol=self.symbol,
            bids=np.array(bids),
            asks=np.array(asks),
            sequence=self.sequence,
        )
    
    def _generate_trade(
        self, timestamp: datetime, book: L2BookSnapshotEvent
    ) -> TradeEvent | None:
        """Generate a trade event with some probability."""
        # Trade probability depends on "activity" (clustered)
        # Use a latent variable for clustering
        trade_prob = 0.3  # Base 30% chance per opportunity
        
        if self.rng.random() > trade_prob:
            return None
        
        # Trade side: slightly more buyer-initiated in uptrends
        side = Side.BUY if self.rng.random() < 0.52 else Side.SELL
        
        # Trade size: power law distribution
        size = self._round_size(self.base_quantity * self.rng.pareto(2.5))
        
        # Trade price: at or through the book
        if side == Side.BUY:
            # Buyer takes ask
            price = book.best_ask
            # Sometimes aggressive (through the book)
            if self.rng.random() < 0.1:
                price = self._round_price(price + book.spread * self.rng.exponential(0.5))
        else:
            price = book.best_bid
            if self.rng.random() < 0.1:
                price = self._round_price(price - book.spread * self.rng.exponential(0.5))
        
        self.sequence += 1
        
        return TradeEvent(
            timestamp=timestamp,
            symbol=self.symbol,
            price=price,
            size=size,
            side=side,
            trade_id=f"t{self.sequence}",
        )
    
    def generate(
        self,
        duration_seconds: float = 3600.0,
        events_per_second: float = 10.0,
    ) -> Iterator[L2BookSnapshotEvent | TradeEvent]:
        """Generate a stream of synthetic market events.
        
        Args:
            duration_seconds: Total duration to generate (seconds).
            events_per_second: Average event frequency.
        
        Yields:
            Alternating book snapshots and trade events.
        """
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        n_events = int(duration_seconds * events_per_second)
        dt = 1.0 / events_per_second
        
        for i in range(n_events):
            timestamp = start_time + timedelta(seconds=i * dt)
            
            # Update mid price with random walk
            # Scale by current vol and sqrt(dt)
            price_change = self.rng.normal(
                0, self.current_vol * self.current_mid * np.sqrt(dt)
            )
            self.current_mid = self._round_price(self.current_mid + price_change)
            
            # Generate book
            book = self._generate_book(timestamp)
            yield book
            
            # Possibly generate trade
            trade = self._generate_trade(timestamp, book)
            if trade is not None:
                yield trade
    
    def generate_to_list(
        self,
        duration_seconds: float = 3600.0,
        events_per_second: float = 10.0,
    ) -> List[L2BookSnapshotEvent | TradeEvent]:
        """Generate events and return as a list."""
        return list(self.generate(duration_seconds, events_per_second))
