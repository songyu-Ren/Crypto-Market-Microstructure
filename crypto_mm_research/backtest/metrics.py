"""Performance metrics for backtest evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from crypto_mm_research.backtest.engine import BacktestResult


@dataclass
class BacktestMetrics:
    """Comprehensive backtest metrics."""
    
    # PnL metrics
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    total_return_pct: float
    
    # Trade metrics
    n_fills: int
    n_buys: int
    n_sells: int
    avg_trade_size: float
    total_volume: float
    
    # Spread capture
    avg_spread_captured: float
    spread_pnl_estimate: float
    
    # Risk metrics
    sharpe_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration: float
    volatility_annual: float
    
    # Inventory metrics
    avg_inventory: float
    max_inventory: float
    inventory_std: float
    
    # Fee metrics
    total_fees: float
    fee_pnl_ratio: float
    
    # Turnover
    turnover: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_pnl": self.total_pnl,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_return_pct": self.total_return_pct,
            "n_fills": self.n_fills,
            "n_buys": self.n_buys,
            "n_sells": self.n_sells,
            "avg_trade_size": self.avg_trade_size,
            "total_volume": self.total_volume,
            "avg_spread_captured": self.avg_spread_captured,
            "spread_pnl_estimate": self.spread_pnl_estimate,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_drawdown_duration": self.max_drawdown_duration,
            "volatility_annual": self.volatility_annual,
            "avg_inventory": self.avg_inventory,
            "max_inventory": self.max_inventory,
            "inventory_std": self.inventory_std,
            "total_fees": self.total_fees,
            "fee_pnl_ratio": self.fee_pnl_ratio,
            "turnover": self.turnover,
        }


def compute_drawdown(equity: pd.Series) -> tuple[np.ndarray, float]:
    """Compute drawdown series and maximum.
    
    Args:
        equity: Equity curve series.
    
    Returns:
        Tuple of (drawdown series, max drawdown).
    """
    if len(equity) < 2:
        return np.array([0.0]), 0.0
    
    rolling_max = equity.expanding().max()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    return drawdown.values, max_dd


def compute_drawdown_duration(equity: pd.Series) -> float:
    """Compute maximum drawdown duration in days."""
    if len(equity) < 2:
        return 0.0
    
    rolling_max = equity.expanding().max()
    in_drawdown = equity < rolling_max
    
    max_duration = 0.0
    current_duration = 0.0
    
    for i in range(len(in_drawdown)):
        if in_drawdown.iloc[i]:
            current_duration += 1
        else:
            max_duration = max(max_duration, current_duration)
            current_duration = 0
    
    max_duration = max(max_duration, current_duration)
    
    # Convert to days (assuming equity index is datetime)
    if hasattr(equity.index, 'freq') and equity.index.freq is not None:
        # If we know frequency
        return max_duration
    
    # Estimate from timestamps
    if len(equity.index) > 1:
        avg_interval = (equity.index[-1] - equity.index[0]).total_seconds() / max(1, len(equity) - 1)
        max_duration_seconds = max_duration * avg_interval
        return max_duration_seconds / (24 * 3600)  # Convert to days
    
    return max_duration


def compute_sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Compute annualized Sharpe ratio.
    
    Args:
        returns: Return series.
        risk_free_rate: Annual risk-free rate.
    
    Returns:
        Annualized Sharpe ratio.
    """
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    
    # Estimate periods per year
    if hasattr(returns.index, 'freq') and returns.index.freq is not None:
        periods_per_year = 252  # Assume daily
    else:
        # Estimate from data
        if len(returns.index) > 1:
            total_seconds = (returns.index[-1] - returns.index[0]).total_seconds()
            periods_per_year = max(1, int(365 * 24 * 3600 / max(1, total_seconds / len(returns))))
        else:
            periods_per_year = 252
    
    excess_returns = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()


def compute_metrics(result: "BacktestResult") -> Dict:
    """Compute comprehensive backtest metrics.
    
    Args:
        result: BacktestResult from engine.
    
    Returns:
        Dictionary of metrics.
    """
    equity_df = result.get_equity_curve()
    fills_df = result.get_fills_df()
    
    if equity_df.empty:
        return _empty_metrics()
    
    # Basic PnL
    initial_equity = result.config.initial_cash
    final_equity = equity_df["total_equity"].iloc[-1]
    total_pnl = final_equity - initial_equity
    total_return_pct = 100.0 * total_pnl / initial_equity
    
    realized_pnl = equity_df["realized_pnl"].iloc[-1]
    unrealized_pnl = equity_df["unrealized_pnl"].iloc[-1]
    
    # Trade metrics
    n_fills = len(result.fill_history)
    n_buys = sum(1 for f in result.fill_history if f.side.value > 0)
    n_sells = n_fills - n_buys
    
    total_volume = sum(f.size for f in result.fill_history)
    avg_trade_size = total_volume / n_fills if n_fills > 0 else 0.0
    
    # Spread capture estimate
    avg_spread_captured = 0.0
    if n_fills >= 2:
        # Estimate from alternating buy/sell
        spread_captures = []
        for i in range(1, len(result.fill_history)):
            f1 = result.fill_history[i - 1]
            f2 = result.fill_history[i]
            if f1.side != f2.side:
                # Round-trip
                capture = abs(f2.price - f1.price)
                spread_captures.append(capture)
        avg_spread_captured = np.mean(spread_captures) if spread_captures else 0.0
    
    spread_pnl_estimate = avg_spread_captured * n_fills / 2 if n_fills > 0 else 0.0
    
    # Risk metrics
    equity = equity_df["total_equity"]
    returns = equity.pct_change().dropna()
    
    _, max_dd = compute_drawdown(equity)
    max_drawdown_pct = 100.0 * abs(max_dd)
    max_drawdown_duration = compute_drawdown_duration(equity)
    
    volatility_annual = returns.std() * np.sqrt(252 * 24 * 12)  # Assume 5-min bars roughly
    sharpe = compute_sharpe(returns)
    
    # Inventory metrics
    inventory = equity_df["position"]
    avg_inventory = inventory.mean()
    max_inventory = inventory.abs().max()
    inventory_std = inventory.std()
    
    # Fee metrics
    total_fees = sum(f.fee for f in result.fill_history)
    fee_pnl_ratio = total_fees / abs(total_pnl) if total_pnl != 0 else 0.0
    
    # Turnover
    turnover = total_volume / initial_equity if initial_equity > 0 else 0.0
    
    return {
        "total_pnl": total_pnl,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "total_return_pct": total_return_pct,
        "n_fills": n_fills,
        "n_buys": n_buys,
        "n_sells": n_sells,
        "avg_trade_size": avg_trade_size,
        "total_volume": total_volume,
        "avg_spread_captured": avg_spread_captured,
        "spread_pnl_estimate": spread_pnl_estimate,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_drawdown_pct,
        "max_drawdown_duration": max_drawdown_duration,
        "volatility_annual": volatility_annual,
        "avg_inventory": avg_inventory,
        "max_inventory": max_inventory,
        "inventory_std": inventory_std,
        "total_fees": total_fees,
        "fee_pnl_ratio": fee_pnl_ratio,
        "turnover": turnover,
    }


def _empty_metrics() -> Dict:
    """Return empty metrics dict."""
    return {
        "total_pnl": 0.0,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "total_return_pct": 0.0,
        "n_fills": 0,
        "n_buys": 0,
        "n_sells": 0,
        "avg_trade_size": 0.0,
        "total_volume": 0.0,
        "avg_spread_captured": 0.0,
        "spread_pnl_estimate": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown_pct": 0.0,
        "max_drawdown_duration": 0.0,
        "volatility_annual": 0.0,
        "avg_inventory": 0.0,
        "max_inventory": 0.0,
        "inventory_std": 0.0,
        "total_fees": 0.0,
        "fee_pnl_ratio": 0.0,
        "turnover": 0.0,
    }
