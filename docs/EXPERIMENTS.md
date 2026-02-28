# Experiments Guide

## Overview

The experiment runner enables batch backtesting with parameter sweeps, automated reporting, and result comparison.

## Configuration

Create a YAML config file:

```yaml
name: "mm_strategy_optimization"

base_config:
  data:
    symbol: "BTC-USDT"
    duration_seconds: 3600
    events_per_second: 10
    volatility_annual: 0.8
    random_seed: 42
  
  strategy:
    target_half_spread_bps: 5.0
    quote_size: 0.1
  
  backtest:
    initial_cash: 100000.0
    maker_fee_rate: 0.0002

parameter_grid:
  "strategy.skew_coeff": [0.0, 1.0, 2.0, 3.0]
  "strategy.inventory_limit": [0.5, 1.0, 2.0]
  "strategy.vol_adaptive": [false, true]

random_seed: 42
```

### Parameter Grid Syntax

- Use dot notation for nested keys: `"strategy.skew_coeff"`
- Lists define the values to sweep
- Cartesian product of all lists = total runs

## Running Experiments

### CLI

```bash
python -m crypto_mm_research.cli.run_experiments \
    --config configs/my_experiment.yaml \
    --output-dir outputs/
```

### Python API

```python
from crypto_mm_research.experiments.runner import ExperimentRunner, ExperimentConfig

config = ExperimentConfig(
    name="my_experiment",
    base_config={...},
    parameter_grid={...},
)

runner = ExperimentRunner(config, output_dir="outputs")
summary = runner.run_all()

top_runs = runner.get_top_runs(n=5, metric="sharpe_ratio")
```

## Output Structure

```
outputs/
  exp_20240115_120000/
    runs/
      run_0000_skew_coeff=0.0_inventory_limit=0.5/
        config_resolved.yaml
        equity_curve.csv
        fills.csv
        metrics.json
      run_0001_.../
        ...
    summary.csv
    report.md
```

## Report Contents

The auto-generated `report.md` includes:

### Summary Statistics
- Mean, std, min, max for key metrics
- PnL, Sharpe, drawdown distributions

### Top Runs
- Top 10 by total PnL
- Key metrics table

### Parameter Sensitivity
- PnL by parameter value
- Grouped statistics

### Risk Analysis
- Drawdown distribution
- PnL vs drawdown correlation

### Adverse Selection
- Spread capture distribution
- Alignment quality stats

## Analysis Tips

### Finding Optimal Parameters

```python
import pandas as pd

summary = pd.read_csv("outputs/exp_*/summary.csv")

# Best by Sharpe
best_sharpe = summary.nlargest(5, "sharpe_ratio")

# Best risk-adjusted (PnL / Drawdown)
summary["risk_adj"] = summary["total_pnl"] / summary["max_drawdown_pct"]
best_risk_adj = summary.nlargest(5, "risk_adj")
```

### Parameter Importance

```python
# Correlation with PnL
correlations = summary[["param_strategy.skew_coeff", "param_strategy.inventory_limit", "total_pnl"]].corr()

# Grouped analysis
summary.groupby("param_strategy.skew_coeff")["total_pnl"].agg(["mean", "std", "count"])
```

### Stability Analysis

```python
# Check if best params are stable across metrics
best_pnl = set(summary.nlargest(10, "total_pnl")["run_name"])
best_sharpe = set(summary.nlargest(10, "sharpe_ratio")["run_name"])

overlap = best_pnl & best_sharpe
print(f"{len(overlap)} runs in both top-10 lists")
```

## Common Experiment Patterns

### Grid Search

```yaml
parameter_grid:
  "strategy.target_half_spread_bps": [3, 5, 7, 10]
  "strategy.skew_coeff": [0, 1, 2, 3]
```

### Ablation Study

```yaml
# Test impact of each feature
parameter_grid:
  "strategy.vol_adaptive": [false, true]
  "execution.queue_ahead_ratio": [0.0, 0.5, 1.0]
```

### Robustness Test

```yaml
# Same params, different random seeds
base_config:
  data:
    random_seed: 42  # Base seed

parameter_grid:
  "data.random_seed": [42, 43, 44, 45, 46]
```

## Best Practices

1. **Start Small**: Test with 2-3 parameters, 2-3 values each
2. **Fix Random Seeds**: Ensure reproducibility
3. **Check Alignment Rate**: Should be >95% for valid results
4. **Use Appropriate Duration**: 1-4 hours for development, 24+ for final validation
5. **Validate Top Results**: Run top 3 configs with longer duration
