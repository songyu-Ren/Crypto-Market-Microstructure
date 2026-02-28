"""CLI entry point for running backtests."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import yaml
import pandas as pd

from crypto_mm_research.data.synthetic import SyntheticDataGenerator
from crypto_mm_research.backtest.engine import BacktestEngine, BacktestConfig
from crypto_mm_research.backtest.strategy import MarketMakingStrategy


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_backtest(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run backtest with configuration."""
    # Extract data config
    data_config = config.get("data", {})
    symbol = data_config.get("symbol", "BTC-USDT")
    duration = data_config.get("duration_seconds", 3600)
    events_per_sec = data_config.get("events_per_second", 10)
    
    # Extract strategy config
    strategy_config = config.get("strategy", {})
    
    # Extract backtest config
    bt_config = config.get("backtest", {})
    initial_cash = bt_config.get("initial_cash", 100000.0)
    maker_fee = bt_config.get("maker_fee_rate", 0.0002)
    taker_fee = bt_config.get("taker_fee_rate", 0.0005)
    
    # Generate synthetic data
    print(f"Generating synthetic data: {symbol}, {duration}s @ {events_per_sec} events/s")
    generator = SyntheticDataGenerator(
        symbol=symbol,
        start_price=data_config.get("start_price", 50000.0),
        tick_size=data_config.get("tick_size", 0.1),
        volatility_annual=data_config.get("volatility_annual", 0.8),
        target_spread_bps=data_config.get("target_spread_bps", 5.0),
        random_seed=data_config.get("random_seed", 42),
    )
    events = generator.generate_to_list(duration, events_per_sec)
    print(f"Generated {len(events)} events")
    
    # Create strategy
    strategy = MarketMakingStrategy(
        target_half_spread_bps=strategy_config.get("target_half_spread_bps", 5.0),
        quote_size=strategy_config.get("quote_size", 0.1),
        skew_coeff=strategy_config.get("skew_coeff", 1.0),
        inventory_limit=strategy_config.get("inventory_limit", 1.0),
        vol_adaptive=strategy_config.get("vol_adaptive", False),
    )
    
    # Create engine
    engine_config = BacktestConfig(
        symbol=symbol,
        initial_cash=initial_cash,
        maker_fee_rate=maker_fee,
        taker_fee_rate=taker_fee,
    )
    engine = BacktestEngine(strategy, engine_config)
    
    # Run backtest
    print("Running backtest...")
    result = engine.run(iter(events))
    
    # Compute metrics
    metrics = result.compute_metrics()
    
    return {
        "result": result,
        "metrics": metrics,
        "config": config,
    }


def print_summary(metrics: Dict[str, Any]) -> None:
    """Print summary table to console."""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS SUMMARY")
    print("=" * 60)
    
    print("\n--- PnL Metrics ---")
    print(f"Total PnL:              ${metrics['total_pnl']:,.2f}")
    print(f"Realized PnL:           ${metrics['realized_pnl']:,.2f}")
    print(f"Unrealized PnL:         ${metrics['unrealized_pnl']:,.2f}")
    print(f"Total Return:           {metrics['total_return_pct']:.3f}%")
    
    print("\n--- Trade Metrics ---")
    print(f"Number of Fills:        {metrics['n_fills']}")
    print(f"Buys:                   {metrics['n_buys']}")
    print(f"Sells:                  {metrics['n_sells']}")
    print(f"Avg Trade Size:         {metrics['avg_trade_size']:.4f}")
    print(f"Total Volume:           {metrics['total_volume']:.4f}")
    
    print("\n--- Spread Capture ---")
    print(f"Avg Spread Captured:    ${metrics['avg_spread_captured']:.4f}")
    print(f"Spread PnL Estimate:    ${metrics['spread_pnl_estimate']:.2f}")
    
    print("\n--- Risk Metrics ---")
    print(f"Sharpe Ratio:           {metrics['sharpe_ratio']:.3f}")
    print(f"Max Drawdown:           {metrics['max_drawdown_pct']:.3f}%")
    print(f"Max DD Duration:        {metrics['max_drawdown_duration']:.2f} days")
    print(f"Volatility (annual):    {metrics['volatility_annual']:.4f}")
    
    print("\n--- Inventory Metrics ---")
    print(f"Avg Inventory:          {metrics['avg_inventory']:.4f}")
    print(f"Max Inventory:          {metrics['max_inventory']:.4f}")
    print(f"Inventory Std:          {metrics['inventory_std']:.4f}")
    
    print("\n--- Fee Metrics ---")
    print(f"Total Fees:             ${metrics['total_fees']:,.2f}")
    print(f"Fee/PnL Ratio:          {metrics['fee_pnl_ratio']:.4f}")
    print(f"Turnover:               {metrics['turnover']:.4f}")
    print("=" * 60)


def save_outputs(
    result: Any,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: str,
) -> None:
    """Save outputs to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save equity curve
    equity_df = result.get_equity_curve()
    equity_path = output_path / "equity_curve.csv"
    equity_df.to_csv(equity_path)
    print(f"\nSaved equity curve to: {equity_path}")
    
    # Save fills
    fills_df = result.get_fills_df()
    fills_path = output_path / "fills.csv"
    fills_df.to_csv(fills_path)
    print(f"Saved fills to: {fills_path}")
    
    # Save metrics as JSON
    metrics_path = output_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Saved metrics to: {metrics_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run crypto market making backtest"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Run backtest
    results = run_backtest(config)
    
    # Print summary
    print_summary(results["metrics"])
    
    # Save outputs
    save_outputs(
        results["result"],
        results["metrics"],
        results["config"],
        args.output_dir,
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
