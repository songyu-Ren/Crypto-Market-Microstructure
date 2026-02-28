# Crypto Market Microstructure + Market Making Backtest

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-quality Python 3.11 research repository for crypto market microstructure analysis and market making backtesting.

## 🎯 Overview

This repository provides a complete framework for researching cryptocurrency market making strategies with a focus on:

- **Market Microstructure**: Order book dynamics, flow imbalance, volatility estimation
- **Realistic Execution**: Queue-based fill models, partial fills, latency simulation
- **Backtesting**: Event-driven simulation with multiple execution models
- **Risk Management**: Kill switches, position limits, drawdown controls
- **Research Rigor**: Leakage prevention, walk-forward validation, experiment management

### Key Features

| Feature | Description |
|---------|-------------|
| 🎲 **Synthetic Data** | Deterministic L2 order book + trade generator |
| 📊 **Feature Engineering** | Microprice, OFI, imbalance, volatility (no lookahead) |
| ⚡ **Execution Models** | Naive and Queue-based with realistic fill dynamics |
| 🌐 **Latency Simulation** | Event time vs arrival time, jitter, out-of-order handling |
| 🧪 **Experiment Runner** | Batch parameter sweeps with automated reporting |
| 🛡️ **Risk Management** | Kill switches, position limits, drawdown controls |
| 📈 **Adapters** | CSV/Parquet loaders for real exchange data |
| 🧪 **Well Tested** | 50+ unit and integration tests |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/songyu-Ren/Crypto-Market-Microstructure.git
cd Crypto-Market-Microstructure

# Create virtual environment and install
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Run Tests

```bash
make test
```

### Run Demo Backtest

```bash
python -m crypto_mm_research.cli.run_backtest \
    --config configs/btcusdt_demo.yaml \
    --output-dir outputs
```

### Run Experiments Grid

```bash
python -m crypto_mm_research.cli.run_experiments \
    --config configs/experiments_grid.yaml \
    --output-dir outputs
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
├── crypto_mm_research/
│   ├── adapters/           # Exchange data adapters
│   ├── backtest/           # Backtest engine
│   ├── data/               # Data models and generators
│   ├── evaluation/         # Leakage control, stability
│   ├── execution/          # Execution models (Naive, Queue)
│   ├── experiments/        # Batch experiment runner
│   ├── features/           # Feature engineering
│   ├── labels/             # Label construction, alignment
│   ├── risk/               # Risk management, kill switches
│   └── cli/                # Command-line interfaces
├── configs/                # YAML configurations
├── data/examples/          # Sample data files
├── notebooks/              # Research notebooks (.py)
├── tests/                  # Test suite (50+ tests)
├── docs/                   # Documentation
├── Makefile
└── pyproject.toml
```

## 🔬 Core Components

### Execution Models

```python
from crypto_mm_research.execution import QueueExecutionModel

# Queue-based execution with realistic fill dynamics
execution_model = QueueExecutionModel(
    tick_size=0.1,
    lot_size=0.001,
    queue_ahead_ratio=0.5,  # Conservative queue position
    allow_partial_fills=True,
    trade_through_enabled=True,
)
```

### Latency Simulation

```python
from crypto_mm_research.execution.latency import LatencyModel, ArrivalTimeGenerator

# Configure latency with jitter
md_latency = LatencyModel(
    base_latency_ms=10,
    jitter_ms=5,
    jitter_type="uniform",
)

# Generate events with arrival times
generator = ArrivalTimeGenerator(
    market_data_latency=md_latency,
    order_latency=LatencyModel(base_latency_ms=20),
    reordering_window_ms=50,
)
```

### Label Construction

```python
from crypto_mm_research.labels import LabelConstructor, LabelConfig

# Construct future return labels (no lookahead)
config = LabelConfig(
    horizons=[1, 5, 10, 30],  # 1s, 5s, 10s, 30s
    direction_threshold_bps=5.0,
)
constructor = LabelConstructor(config)
labels_df = constructor.construct_labels(mid_prices, timestamps)
```

### Experiment Runner

```python
from crypto_mm_research.experiments import ExperimentRunner, ExperimentConfig

config = ExperimentConfig(
    name="spread_optimization",
    base_config={...},
    parameter_grid={
        "strategy.target_half_spread_bps": [3, 5, 7],
        "strategy.skew_coeff": [0, 1, 2],
    },
)

runner = ExperimentRunner(config, output_dir="outputs")
summary = runner.run_all()
top_runs = runner.get_top_runs(n=5, metric="sharpe_ratio")
```

### Risk Management

```python
from crypto_mm_research.risk import RiskManager, RiskConfig

risk_config = RiskConfig(
    max_position_absolute=10.0,
    max_drawdown_pct=5.0,
    drawdown_kill_switch=True,
)
risk_manager = RiskManager(risk_config)

# Check before placing orders
if not risk_manager.check_position_limit(position_size, notional, timestamp):
    # Block order
    pass
```

## 📊 Performance Metrics

- **PnL Metrics**: Total, realized, unrealized, return %
- **Trade Stats**: Fills, volume, avg size, queue position
- **Spread Capture**: Estimated vs actual
- **Adverse Selection**: t+H mid - fill mid
- **Risk**: Sharpe, max drawdown, volatility
- **Alignment**: Fill-to-mid alignment rate

## 🛡️ Design Principles

### No Lookahead Guarantee

All features use only information available at time `t`:

```python
# Rolling features use shift(1)
rolling_mean = df["price"].shift(1).rolling(window=20).mean()

# Labels are shifted forward
labels = labels.shift(-1)  # Align with features at time t
```

### Event Time vs Arrival Time

```python
# Event time: When event occurred in market
event_ts = book.timestamp

# Arrival time: When event reaches our system
arrival_ts = event_ts + latency + jitter

# Engine processes by arrival_ts
# Metrics computed using event_ts (no lookahead)
```

### Determinism

Same random seed → identical results:

```python
gen = SyntheticDataGenerator(random_seed=42)
events = gen.generate_to_list(...)  # Always same data
```

## 📚 Documentation

- [EXECUTION_MODEL.md](docs/EXECUTION_MODEL.md) - Queue mechanics, latency, fills
- [EXPERIMENTS.md](docs/EXPERIMENTS.md) - Batch experiments and reporting
- [LEAKAGE_AND_STABILITY.md](docs/LEAKAGE_AND_STABILITY.md) - Preventing lookahead bias
- [METRICS.md](docs/METRICS.md) - Precise metric definitions

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage
pytest tests/ --cov=crypto_mm_research --cov-report=html

# Run specific test file
pytest tests/unit/test_execution.py -v
```

## 🔧 Development

```bash
# Format code
make format

# Run linting
make lint

# Run all checks
make all
```

## 📖 Citation

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

**Disclaimer**: This software is for research and educational purposes only. Past performance does not guarantee future results.
