# Crypto Market Microstructure + Market Making Backtest

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-quality Python 3.11 research repository for crypto market microstructure analysis and market making backtesting.

## 🎯 Overview

This repository provides a complete framework for researching cryptocurrency market making strategies with a focus on:

- **Market Microstructure**: Order book dynamics, flow imbalance, volatility estimation
- **Backtesting**: Event-driven simulation with realistic fill models
- **Risk Management**: Inventory skew, drawdown control, regime analysis
- **Research Rigor**: Leakage prevention, walk-forward validation, stability checks

### Key Features

| Feature | Description |
|---------|-------------|
| 🎲 **Synthetic Data** | Deterministic L2 order book + trade generator for offline research |
| 📊 **Feature Engineering** | Microprice, OFI, depth imbalance, realized volatility (no lookahead) |
| ⚡ **Event-Driven Engine** | Realistic fill model with maker/taker fees |
| 📈 **Strategy Framework** | Baseline MM with inventory skew + vol-adaptive spreads |
| 🛡️ **Leakage Control** | Timestamp discipline, walk-forward splits, regime analysis |
| 🧪 **Well Tested** | 80+ unit and integration tests |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/songyu-Ren/Crypto-Market-Microstructure.git
cd Crypto-Market-Microstructure

# Create virtual environment and install
make install-dev

# Or manually
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Run Demo Backtest

```bash
python -m crypto_mm_research.cli.run_backtest \
    --config configs/btcusdt_demo.yaml \
    --output-dir outputs
```

Sample output:
```
============================================================
BACKTEST RESULTS SUMMARY
============================================================

--- PnL Metrics ---
Total PnL:              $125.43
Realized PnL:           $98.21
Unrealized PnL:         $27.22
Total Return:           0.125%

--- Trade Metrics ---
Number of Fills:        156
Avg Trade Size:         0.1000
Total Volume:           15.6000

--- Risk Metrics ---
Sharpe Ratio:           1.234
Max Drawdown:           0.089%
Volatility (annual):    0.2341

--- Inventory Metrics ---
Avg Inventory:          0.0234
Max Inventory:          0.4500
```

### Run Research Notebooks

```bash
# Order book features analysis
python notebooks/01_orderbook_features_demo.py

# Market making strategy comparison
python notebooks/02_market_making_backtest_demo.py
```

## 📁 Project Structure

```
crypto_mm_research/
├── crypto_mm_research/          # Main package
│   ├── data/                    # Data models and loaders
│   │   ├── events.py            # L2BookSnapshotEvent, TradeEvent
│   │   ├── synthetic.py         # SyntheticDataGenerator
│   │   └── loader.py            # CSV/Parquet data loaders
│   ├── features/                # Feature engineering
│   │   ├── microstructure.py    # Microstructure features
│   │   └── builder.py           # FeatureBuilder (sequential)
│   ├── backtest/                # Backtest engine
│   │   ├── engine.py            # Event-driven backtester
│   │   ├── strategy.py          # Strategy interface
│   │   ├── account.py           # PnL accounting
│   │   └── metrics.py           # Performance metrics
│   ├── evaluation/              # Evaluation utilities
│   │   ├── leakage.py           # Lookahead prevention
│   │   └── stability.py         # Walk-forward, regimes
│   └── cli/                     # Command-line interface
│       └── run_backtest.py
├── configs/                     # YAML configurations
├── notebooks/                   # Research notebooks (.py)
├── tests/                       # Test suite (80+ tests)
├── docs/                        # Documentation
├── Makefile                     # Build automation
└── pyproject.toml               # Package configuration
```

## 🔬 Core Components

### 1. Data Model

```python
from crypto_mm_research.data.events import L2BookSnapshotEvent, TradeEvent
from crypto_mm_research.data.synthetic import SyntheticDataGenerator

# Generate deterministic synthetic data
gen = SyntheticDataGenerator(
    symbol="BTC-USDT",
    start_price=50000.0,
    volatility_annual=0.8,
    random_seed=42
)
events = gen.generate_to_list(duration_seconds=3600, events_per_second=10)
```

### 2. Feature Engineering

```python
from crypto_mm_research.features.builder import FeatureBuilder

# Build features sequentially (no lookahead)
builder = FeatureBuilder(symbol="BTC-USDT", window_seconds=20.0)
features_df = builder.process_events(events)

# Available features:
# - mid_price, microprice, spread_bps
# - book_imbalance, depth_imbalance_5
# - ofi (order flow imbalance)
# - realized_vol_20, zscore_mid_20
```

### 3. Backtest Engine

```python
from crypto_mm_research.backtest.engine import BacktestEngine, BacktestConfig
from crypto_mm_research.backtest.strategy import MarketMakingStrategy

# Configure strategy
strategy = MarketMakingStrategy(
    target_half_spread_bps=5.0,
    quote_size=0.1,
    skew_coeff=2.0,          # Inventory skew
    inventory_limit=1.0,
    vol_adaptive=True
)

# Run backtest
config = BacktestConfig(initial_cash=100000.0)
engine = BacktestEngine(strategy, config)
result = engine.run(iter(events))

# Analyze results
metrics = result.compute_metrics()
equity_df = result.get_equity_curve()
```

### 4. Leakage Control

```python
from crypto_mm_research.evaluation.leakage import (
    validate_no_lookahead,
    assert_no_overlap
)
from crypto_mm_research.evaluation.stability import walk_forward_split

# Validate no lookahead
validate_no_lookahead(features, target, target_lag=timedelta(seconds=1))

# Walk-forward validation
for train_data, test_data in walk_forward_split(
    data, n_splits=5, gap=timedelta(minutes=5)
):
    model.fit(train_data)
    predictions = model.predict(test_data)
```

## 📊 Performance Metrics

| Metric | Description |
|--------|-------------|
| **Total PnL** | Final equity - initial cash |
| **Realized PnL** | Closed trades PnL net of fees |
| **Sharpe Ratio** | Risk-adjusted return (annualized) |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Spread Capture** | Estimated spread PnL vs actual |
| **Inventory Metrics** | Avg, max, std of position |
| **Fee/PnL Ratio** | Fees as fraction of gross PnL |
| **Turnover** | Volume / capital |

## 🛡️ Design Principles

### No Lookahead Guarantee

All features use only information available at time `t`:

```python
# Rolling features use shift(1) to exclude current value
rolling_mean = df["price"].shift(1).rolling(window=20).mean()

# Z-scores compute stats on past data only
mean = np.mean(values[:-1])  # Exclude current
std = np.std(values[:-1], ddof=1)
```

### Determinism

Same random seed → identical results:

```python
gen = SyntheticDataGenerator(random_seed=42)
events = gen.generate_to_list(...)  # Always same data
```

### Realistic Fill Model

- Bid fills if `bid >= best_ask` OR trade at/through price
- Ask fills if `ask <= best_bid` OR trade at/through price
- Maker fees applied to all fills

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage
pytest tests/ --cov=crypto_mm_research --cov-report=html

# Run specific test file
pytest tests/unit/test_backtest.py -v
```

## 🔧 Development

```bash
# Format code
make format

# Run linting (ruff + mypy)
make lint

# Run all checks
make all
```

## 📚 Documentation

- [LEAKAGE_AND_STABILITY.md](docs/LEAKAGE_AND_STABILITY.md) - Guide to preventing lookahead bias
- [METRICS.md](docs/METRICS.md) - Precise definitions of all metrics

## 🎓 Research Applications

1. **Market Making**: Test inventory skew, spread capture vs adverse selection
2. **Microstructure**: Analyze order flow, book dynamics, price impact
3. **Strategy Development**: Rapid prototyping with synthetic data
4. **Risk Analysis**: Drawdown, regime performance, stability metrics

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@software{crypto_mm_research,
  author = {Songyu Ren},
  title = {Crypto Market Microstructure + Market Making Backtest},
  url = {https://github.com/songyu-Ren/Crypto-Market-Microstructure},
  year = {2024}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Cont et al. (2014)](https://arxiv.org/abs/1312.6403) - Order flow imbalance
- [Lopez de Prado (2018)](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089) - Financial ML best practices
- [Avellaneda & Stoikov (2008)](https://math.nyu.edu/~avellane/HighFrequencyTrading.pdf) - Market making theory

---

**Disclaimer**: This software is for research and educational purposes only. It is not intended for live trading without extensive testing and validation. Past performance (even in backtests) does not guarantee future results.
