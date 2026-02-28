"""Microstructure feature computation functions.

All functions operate on data available up to time t (no lookahead).
"""

from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd

from crypto_mm_research.data.events import L2BookSnapshotEvent, TradeEvent


def compute_mid_price(book: L2BookSnapshotEvent) -> float:
    """Compute mid price from best bid and ask.
    
    Args:
        book: L2 book snapshot.
    
    Returns:
        Mid price, or NaN if book is empty.
    """
    return book.mid_price


def compute_spread(book: L2BookSnapshotEvent) -> float:
    """Compute bid-ask spread.
    
    Args:
        book: L2 book snapshot.
    
    Returns:
        Spread in price terms, or NaN if book is empty.
    """
    return book.spread


def compute_spread_bps(book: L2BookSnapshotEvent) -> float:
    """Compute bid-ask spread in basis points.
    
    Args:
        book: L2 book snapshot.
    
    Returns:
        Spread in basis points.
    """
    mid = book.mid_price
    if mid == 0:
        return 0.0
    return 10000.0 * book.spread / mid


def compute_microprice(book: L2BookSnapshotEvent) -> float:
    """Compute volume-weighted microprice.
    
    Microprice gives more weight to the side with less volume,
    indicating where the price is more likely to move.
    
    Args:
        book: L2 book snapshot.
    
    Returns:
        Microprice value.
    """
    return book.microprice


def compute_book_imbalance(book: L2BookSnapshotEvent) -> float:
    """Compute top-of-book imbalance.
    
    Imbalance = (bid_size - ask_size) / (bid_size + ask_size)
    Range: [-1, 1], positive means more bid volume.
    
    Args:
        book: L2 book snapshot.
    
    Returns:
        Imbalance ratio.
    """
    if book.bids.size == 0 or book.asks.size == 0:
        return 0.0
    
    bid_size = book.bids[0, 1]
    ask_size = book.asks[0, 1]
    
    total = bid_size + ask_size
    if total == 0:
        return 0.0
    
    return (bid_size - ask_size) / total


def compute_depth_imbalance(
    book: L2BookSnapshotEvent, levels: int = 5
) -> float:
    """Compute depth-weighted imbalance across k levels.
    
    Weights decrease linearly with depth (level 1 has weight k,
    level k has weight 1).
    
    Args:
        book: L2 book snapshot.
        levels: Number of levels to consider.
    
    Returns:
        Depth-weighted imbalance.
    """
    bid_vol = 0.0
    ask_vol = 0.0
    
    for i in range(min(levels, book.bids.shape[0])):
        weight = levels - i
        bid_vol += weight * book.bids[i, 1]
    
    for i in range(min(levels, book.asks.shape[0])):
        weight = levels - i
        ask_vol += weight * book.asks[i, 1]
    
    total = bid_vol + ask_vol
    if total == 0:
        return 0.0
    
    return (bid_vol - ask_vol) / total


def compute_ofi(
    prev_book: L2BookSnapshotEvent | None,
    curr_book: L2BookSnapshotEvent,
    trades: List[TradeEvent],
) -> float:
    """Compute Order Flow Imbalance (OFI).
    
    OFI measures the net order flow at the top of book.
    Based on Cont et al. (2014) and variations.
    
    This implementation computes OFI as:
    - Change in bid volume at best bid (if bid price unchanged or up)
    - Minus change in ask volume at best ask (if ask price unchanged or down)
    - Plus aggressive buy volume (trades at/above ask)
    - Minus aggressive sell volume (trades at/below bid)
    
    Args:
        prev_book: Previous book snapshot (None for first).
        curr_book: Current book snapshot.
        trades: List of trades since previous book.
    
    Returns:
        OFI value (positive = buying pressure).
    """
    if prev_book is None:
        return 0.0
    
    ofi = 0.0
    
    # Book changes at top level
    if curr_book.best_bid > prev_book.best_bid:
        # Bid moved up - all new bid volume is positive OFI
        if curr_book.bids.size > 0:
            ofi += curr_book.bids[0, 1]
    elif curr_book.best_bid == prev_book.best_bid:
        # Bid unchanged - delta in volume
        if curr_book.bids.size > 0 and prev_book.bids.size > 0:
            ofi += curr_book.bids[0, 1] - prev_book.bids[0, 1]
    else:
        # Bid moved down - lost volume is negative OFI
        if prev_book.bids.size > 0:
            ofi -= prev_book.bids[0, 1]
    
    if curr_book.best_ask < prev_book.best_ask:
        # Ask moved down - all new ask volume is negative OFI
        if curr_book.asks.size > 0:
            ofi -= curr_book.asks[0, 1]
    elif curr_book.best_ask == prev_book.best_ask:
        # Ask unchanged - delta in volume
        if curr_book.asks.size > 0 and prev_book.asks.size > 0:
            ofi -= curr_book.asks[0, 1] - prev_book.asks[0, 1]
    else:
        # Ask moved up - lost volume is positive OFI
        if prev_book.asks.size > 0:
            ofi += prev_book.asks[0, 1]
    
    # Add trade aggressiveness
    for trade in trades:
        if trade.side.value > 0:  # Buyer-initiated
            ofi += trade.size
        else:  # Seller-initiated
            ofi -= trade.size
    
    return ofi


def compute_realized_volatility(
    returns: List[float],
    window: int = 20,
) -> float:
    """Compute short-horizon realized volatility estimate.
    
    Uses the square root of sum of squared returns.
    
    Args:
        returns: List of recent returns.
        window: Window size for calculation.
    
    Returns:
        Realized volatility (standard deviation of returns).
    """
    if len(returns) < 2:
        return 0.0
    
    recent = returns[-window:] if len(returns) >= window else returns
    if len(recent) < 2:
        return 0.0
    
    return float(np.std(recent, ddof=1))


def compute_realized_variance(
    returns: List[float],
    window: int = 20,
) -> float:
    """Compute realized variance (squared volatility).
    
    Args:
        returns: List of recent returns.
        window: Window size.
    
    Returns:
        Realized variance.
    """
    if len(returns) < 2:
        return 0.0
    
    recent = returns[-window:] if len(returns) >= window else returns
    if len(recent) < 2:
        return 0.0
    
    return float(np.var(recent, ddof=1))


def compute_rolling_zscore(
    values: List[float],
    window: int = 20,
) -> float:
    """Compute z-score of the most recent value.
    
    Args:
        values: List of values.
        window: Rolling window size.
    
    Returns:
        Z-score (deviation from mean in standard deviations).
    """
    if len(values) < 2:
        return 0.0
    
    recent = values[-window:] if len(values) >= window else values
    if len(recent) < 2:
        return 0.0
    
    mean = np.mean(recent[:-1])  # Mean excluding current (no lookahead)
    std = np.std(recent[:-1], ddof=1)
    
    if std == 0:
        return 0.0
    
    return (values[-1] - mean) / std


def compute_price_momentum(
    prices: List[float],
    window: int = 20,
) -> float:
    """Compute price momentum (return over window).
    
    Args:
        prices: List of prices.
        window: Lookback window.
    
    Returns:
        Momentum as return (not percentage).
    """
    if len(prices) < window + 1:
        return 0.0
    
    prev_price = prices[-window - 1]
    curr_price = prices[-1]
    
    if prev_price == 0:
        return 0.0
    
    return (curr_price - prev_price) / prev_price


def compute_trade_imbalance(
    trades: List[TradeEvent],
    window_seconds: float = 60.0,
) -> float:
    """Compute trade imbalance over recent window.
    
    Args:
        trades: List of trades.
        window_seconds: Time window in seconds.
    
    Returns:
        Trade imbalance ratio [-1, 1].
    """
    if not trades:
        return 0.0
    
    buy_vol = sum(t.size for t in trades if t.side.value > 0)
    sell_vol = sum(t.size for t in trades if t.side.value < 0)
    total = buy_vol + sell_vol
    
    if total == 0:
        return 0.0
    
    return (buy_vol - sell_vol) / total


def compute_trade_intensity(
    trades: List[TradeEvent],
    window_seconds: float = 60.0,
) -> float:
    """Compute trade intensity (trades per second).
    
    Args:
        trades: List of trades.
        window_seconds: Time window.
    
    Returns:
        Trade intensity (trades per second).
    """
    if not trades or len(trades) < 2:
        return 0.0
    
    # Count trades in window
    cutoff = trades[-1].timestamp - pd.Timedelta(seconds=window_seconds)
    recent_trades = [t for t in trades if t.timestamp >= cutoff]
    
    if len(recent_trades) < 2:
        return 0.0
    
    duration = (recent_trades[-1].timestamp - recent_trades[0].timestamp).total_seconds()
    if duration <= 0:
        return 0.0
    
    return len(recent_trades) / duration


def compute_vwap(trades: List[TradeEvent]) -> float:
    """Compute volume-weighted average price from trades.
    
    Args:
        trades: List of trades.
    
    Returns:
        VWAP, or 0 if no trades.
    """
    if not trades:
        return 0.0
    
    total_value = sum(t.price * t.size for t in trades)
    total_volume = sum(t.size for t in trades)
    
    if total_volume == 0:
        return 0.0
    
    return total_value / total_volume
