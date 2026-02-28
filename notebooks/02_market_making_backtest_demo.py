"""
Notebook 02: Market Making Backtest Demo
========================================

This notebook demonstrates:
1. Running a market making backtest
2. Analyzing PnL components
3. Visualizing equity curve and fills
4. Comparing different strategy configurations

Run with: python notebooks/02_market_making_backtest_demo.py
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from crypto_mm_research.data.synthetic import SyntheticDataGenerator
from crypto_mm_research.backtest.engine import BacktestEngine, BacktestConfig
from crypto_mm_research.backtest.strategy import MarketMakingStrategy


def generate_data(duration_seconds=3600, random_seed=42):
    """Generate synthetic market data."""
    print(f"Generating {duration_seconds}s of synthetic data...")
    
    generator = SyntheticDataGenerator(
        symbol="BTC-USDT",
        start_price=50000.0,
        tick_size=0.1,
        volatility_annual=0.8,
        target_spread_bps=5.0,
        book_levels=10,
        random_seed=random_seed,
    )
    
    events = generator.generate_to_list(
        duration_seconds=duration_seconds,
        events_per_second=10,
    )
    
    print(f"Generated {len(events)} events")
    return events


def run_backtest(events, strategy_params, label="Strategy"):
    """Run a single backtest."""
    print(f"\nRunning backtest: {label}")
    
    strategy = MarketMakingStrategy(**strategy_params)
    
    config = BacktestConfig(
        symbol="BTC-USDT",
        initial_cash=100000.0,
        maker_fee_rate=0.0002,
        taker_fee_rate=0.0005,
        record_interval_seconds=1.0,
    )
    
    engine = BacktestEngine(strategy, config)
    result = engine.run(iter(events))
    
    metrics = result.compute_metrics()
    
    print(f"  Total PnL: ${metrics['total_pnl']:.2f}")
    print(f"  Return: {metrics['total_return_pct']:.3f}%")
    print(f"  Sharpe: {metrics['sharpe_ratio']:.3f}")
    print(f"  Max DD: {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Fills: {metrics['n_fills']}")
    print(f"  Avg Inventory: {metrics['avg_inventory']:.4f}")
    
    return result, metrics


def plot_equity_comparison(results_dict, output_path="outputs/02_equity_comparison.png"):
    """Plot equity curves for multiple strategies."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    colors = ["blue", "green", "red", "orange"]
    
    for i, (label, result) in enumerate(results_dict.items()):
        equity_df = result.get_equity_curve()
        if equity_df.empty:
            continue
        
        color = colors[i % len(colors)]
        
        # Equity curve
        axes[0].plot(
            equity_df.index,
            equity_df["total_equity"],
            label=label,
            color=color,
            alpha=0.8,
        )
        
        # Inventory
        axes[1].plot(
            equity_df.index,
            equity_df["position"],
            label=label,
            color=color,
            alpha=0.7,
        )
    
    axes[0].set_ylabel("Equity ($)")
    axes[0].set_title("Equity Curve Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_ylabel("Position")
    axes[1].set_xlabel("Time")
    axes[1].set_title("Inventory Over Time")
    axes[1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved plot to {output_path}")
    plt.close()


def plot_pnl_components(result, output_path="outputs/02_pnl_components.png"):
    """Plot PnL components over time."""
    equity_df = result.get_equity_curve()
    if equity_df.empty:
        print("No equity data to plot")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Total equity
    axes[0].plot(equity_df.index, equity_df["total_equity"], color="blue")
    axes[0].axhline(y=100000, color="black", linestyle="--", alpha=0.5, label="Start")
    axes[0].set_ylabel("Equity ($)")
    axes[0].set_title("Total Equity")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Realized vs Unrealized
    axes[1].plot(equity_df.index, equity_df["realized_pnl"], label="Realized", color="green")
    axes[1].plot(equity_df.index, equity_df["unrealized_pnl"], label="Unrealized", color="orange")
    axes[1].set_ylabel("PnL ($)")
    axes[1].set_title("Realized vs Unrealized PnL")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Cash
    axes[2].plot(equity_df.index, equity_df["cash"], color="purple")
    axes[2].set_ylabel("Cash ($)")
    axes[2].set_xlabel("Time")
    axes[2].set_title("Cash Balance")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_fill_analysis(result, output_path="outputs/02_fill_analysis.png"):
    """Analyze fill patterns."""
    fills_df = result.get_fills_df()
    if fills_df.empty:
        print("No fills to analyze")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Fill prices over time
    buy_fills = fills_df[fills_df["side"] == "buy"]
    sell_fills = fills_df[fills_df["side"] == "sell"]
    
    axes[0, 0].scatter(
        buy_fills.index,
        buy_fills["price"],
        color="green",
        alpha=0.6,
        s=20,
        label="Buy",
    )
    axes[0, 0].scatter(
        sell_fills.index,
        sell_fills["price"],
        color="red",
        alpha=0.6,
        s=20,
        label="Sell",
    )
    axes[0, 0].set_ylabel("Price ($)")
    axes[0, 0].set_title("Fill Prices")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Fill sizes
    axes[0, 1].hist(fills_df["size"], bins=30, color="blue", alpha=0.7, edgecolor="black")
    axes[0, 1].set_xlabel("Size")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Fill Size Distribution")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cumulative fills
    buy_cumsum = (buy_fills["size"] * buy_fills["price"]).cumsum() if not buy_fills.empty else pd.Series()
    sell_cumsum = (sell_fills["size"] * sell_fills["price"]).cumsum() if not sell_fills.empty else pd.Series()
    
    if not buy_cumsum.empty:
        axes[1, 0].plot(buy_cumsum.index, buy_cumsum.values, label="Buy Notional", color="green")
    if not sell_cumsum.empty:
        axes[1, 0].plot(sell_cumsum.index, sell_cumsum.values, label="Sell Notional", color="red")
    axes[1, 0].set_ylabel("Cumulative Notional ($)")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_title("Cumulative Fill Notional")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Fees
    axes[1, 1].hist(fills_df["fee"], bins=30, color="orange", alpha=0.7, edgecolor="black")
    axes[1, 1].set_xlabel("Fee ($)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Fee Distribution")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_drawdown(result, output_path="outputs/02_drawdown.png"):
    """Plot drawdown analysis."""
    equity_df = result.get_equity_curve()
    if equity_df.empty:
        return
    
    equity = equity_df["total_equity"]
    rolling_max = equity.expanding().max()
    drawdown = (equity - rolling_max) / rolling_max * 100  # In percent
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Equity and peak
    axes[0].plot(equity.index, equity, label="Equity", color="blue")
    axes[0].plot(rolling_max.index, rolling_max, label="Peak", color="green", linestyle="--")
    axes[0].set_ylabel("Equity ($)")
    axes[0].set_title("Equity vs Peak")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Drawdown
    axes[1].fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.3)
    axes[1].plot(drawdown.index, drawdown, color="red", linewidth=1)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Time")
    axes[1].set_title("Drawdown from Peak")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    plt.close()


def print_comparison_table(metrics_dict):
    """Print comparison table of strategies."""
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON")
    print("=" * 80)
    
    # Header
    header = f"{'Metric':<25}"
    for label in metrics_dict.keys():
        header += f"{label:<20}"
    print(header)
    print("-" * 80)
    
    # Metrics to compare
    metrics_to_show = [
        ("Total PnL ($)", "total_pnl", "${:.2f}"),
        ("Return (%)", "total_return_pct", "{:.3f}%"),
        ("Sharpe Ratio", "sharpe_ratio", "{:.3f}"),
        ("Max DD (%)", "max_drawdown_pct", "{:.2f}%"),
        ("Fills", "n_fills", "{:d}"),
        ("Avg Inventory", "avg_inventory", "{:.4f}"),
        ("Max Inventory", "max_inventory", "{:.4f}"),
        ("Total Fees ($)", "total_fees", "${:.2f}"),
        ("Turnover", "turnover", "{:.4f}"),
    ]
    
    for metric_name, metric_key, fmt in metrics_to_show:
        row = f"{metric_name:<25}"
        for label, metrics in metrics_dict.items():
            value = metrics.get(metric_key, 0)
            row += f"{fmt.format(value):<20}"
        print(row)
    
    print("=" * 80)


def main():
    """Main notebook execution."""
    print("=" * 80)
    print("NOTEBOOK 02: Market Making Backtest Demo")
    print("=" * 80)
    
    # Generate data once for fair comparison
    events = generate_data(duration_seconds=3600, random_seed=42)
    
    # Define strategies to compare
    strategies = {
        "Baseline (no skew)": {
            "target_half_spread_bps": 5.0,
            "quote_size": 0.1,
            "skew_coeff": 0.0,  # No inventory skew
            "inventory_limit": 10.0,
        },
        "With Inventory Skew": {
            "target_half_spread_bps": 5.0,
            "quote_size": 0.1,
            "skew_coeff": 2.0,  # Moderate skew
            "inventory_limit": 1.0,
        },
        "Tight Spread + Skew": {
            "target_half_spread_bps": 3.0,
            "quote_size": 0.1,
            "skew_coeff": 3.0,  # Strong skew
            "inventory_limit": 0.5,
        },
    }
    
    # Run backtests
    results = {}
    metrics = {}
    
    for label, params in strategies.items():
        result, m = run_backtest(events, params, label)
        results[label] = result
        metrics[label] = m
    
    # Print comparison
    print_comparison_table(metrics)
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Use "With Inventory Skew" for detailed analysis
    primary_result = results["With Inventory Skew"]
    
    plot_equity_comparison(results)
    plot_pnl_components(primary_result)
    plot_fill_analysis(primary_result)
    plot_drawdown(primary_result)
    
    # Save results
    for label, result in results.items():
        safe_label = label.replace(" ", "_").replace("(", "").replace(")", "").lower()
        equity_path = f"outputs/02_equity_{safe_label}.csv"
        result.get_equity_curve().to_csv(equity_path)
        print(f"Saved equity curve for {label} to {equity_path}")
    
    print("\n" + "=" * 80)
    print("Notebook 02 complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
