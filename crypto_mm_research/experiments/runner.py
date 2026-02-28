"""Experiment runner for batch backtests with parameter grids."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Iterator
import itertools
import json
import yaml

import pandas as pd
import numpy as np

from crypto_mm_research.data.synthetic import SyntheticDataGenerator
from crypto_mm_research.backtest.engine import BacktestEngine, BacktestConfig
from crypto_mm_research.backtest.strategy import MarketMakingStrategy


@dataclass
class ExperimentConfig:
    """Configuration for experiment runner."""
    
    name: str
    base_config: Dict[str, Any]
    parameter_grid: Dict[str, List[Any]]
    random_seed: int = 42
    
    def generate_runs(self) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Generate all parameter combinations."""
        if not self.parameter_grid:
            yield "run_0", self.base_config.copy()
            return
        
        keys = list(self.parameter_grid.keys())
        values = [self.parameter_grid[k] for k in keys]
        
        for i, combo in enumerate(itertools.product(*values)):
            run_config = self.base_config.copy()
            run_name = f"run_{i:04d}"
            
            for key, value in zip(keys, combo):
                parts = key.split(".")
                target = run_config
                for part in parts[:-1]:
                    if part not in target:
                        target[part] = {}
                    target = target[part]
                target[parts[-1]] = value
                
                run_name += f"_{parts[-1]}={value}"
            
            yield run_name, run_config


class ExperimentRunner:
    """Run batch experiments with parameter sweeps."""
    
    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: str = "outputs",
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir) / f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[Dict[str, Any]] = []
    
    def run_all(self, verbose: bool = True) -> pd.DataFrame:
        """Run all experiments and return summary."""
        runs_dir = self.output_dir / "runs"
        runs_dir.mkdir(exist_ok=True)
        
        for run_name, run_config in self.config.generate_runs():
            if verbose:
                print(f"\n{'='*60}")
                print(f"Running: {run_name}")
                print(f"{'='*60}")
            
            result = self._run_single(run_name, run_config, runs_dir)
            self.results.append(result)
        
        summary_df = self._generate_summary()
        summary_path = self.output_dir / "summary.csv"
        summary_df.to_csv(summary_path)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Experiment complete. Summary saved to: {summary_path}")
            print(f"{'='*60}")
        
        return summary_df
    
    def _run_single(
        self,
        run_name: str,
        run_config: Dict[str, Any],
        runs_dir: Path,
    ) -> Dict[str, Any]:
        """Run single experiment."""
        run_dir = runs_dir / run_name
        run_dir.mkdir(exist_ok=True)
        
        config_path = run_dir / "config_resolved.yaml"
        with open(config_path, "w") as f:
            yaml.dump(run_config, f)
        
        data_config = run_config.get("data", {})
        generator = SyntheticDataGenerator(
            symbol=data_config.get("symbol", "BTC-USDT"),
            start_price=data_config.get("start_price", 50000.0),
            tick_size=data_config.get("tick_size", 0.1),
            volatility_annual=data_config.get("volatility_annual", 0.8),
            target_spread_bps=data_config.get("target_spread_bps", 5.0),
            random_seed=data_config.get("random_seed", self.config.random_seed),
        )
        
        events = generator.generate_to_list(
            duration_seconds=data_config.get("duration_seconds", 3600),
            events_per_second=data_config.get("events_per_second", 10),
        )
        
        strategy_config = run_config.get("strategy", {})
        strategy = MarketMakingStrategy(
            target_half_spread_bps=strategy_config.get("target_half_spread_bps", 5.0),
            quote_size=strategy_config.get("quote_size", 0.1),
            skew_coeff=strategy_config.get("skew_coeff", 1.0),
            inventory_limit=strategy_config.get("inventory_limit", 1.0),
            vol_adaptive=strategy_config.get("vol_adaptive", False),
        )
        
        bt_config = run_config.get("backtest", {})
        engine_config = BacktestConfig(
            symbol=data_config.get("symbol", "BTC-USDT"),
            initial_cash=bt_config.get("initial_cash", 100000.0),
            maker_fee_rate=bt_config.get("maker_fee_rate", 0.0002),
            taker_fee_rate=bt_config.get("taker_fee_rate", 0.0005),
        )
        
        engine = BacktestEngine(strategy, engine_config)
        result = engine.run(iter(events))
        
        metrics = result.compute_metrics()
        
        equity_df = result.get_equity_curve()
        equity_path = run_dir / "equity_curve.csv"
        equity_df.to_csv(equity_path)
        
        fills_df = result.get_fills_df()
        fills_path = run_dir / "fills.csv"
        fills_df.to_csv(fills_path)
        
        metrics_path = run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        
        result_summary = {
            "run_name": run_name,
            "run_dir": str(run_dir),
            **{f"param_{k}": v for k, v in self._flatten_config(run_config).items()},
            **metrics,
        }
        
        return result_summary
    
    def _flatten_config(self, config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested config dict."""
        flat = {}
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(self._flatten_config(value, full_key))
            else:
                flat[full_key] = value
        return flat
    
    def _generate_summary(self) -> pd.DataFrame:
        """Generate summary DataFrame."""
        if not self.results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results)
        
        if "total_pnl" in df.columns:
            df = df.sort_values("total_pnl", ascending=False)
        
        return df
    
    def get_top_runs(self, n: int = 5, metric: str = "total_pnl") -> pd.DataFrame:
        """Get top N runs by metric."""
        summary = self._generate_summary()
        if metric not in summary.columns:
            return pd.DataFrame()
        
        return summary.nlargest(n, metric)
