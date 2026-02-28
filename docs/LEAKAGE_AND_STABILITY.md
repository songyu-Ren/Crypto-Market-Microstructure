# Leakage Control and Stability

## What is Leakage in Time Series?

**Leakage** (or lookahead bias) occurs when information from the future is used to make decisions at time t. In market microstructure research, this is particularly dangerous because:

1. **Feature Leakage**: Features computed at time t implicitly include data from t+1, t+2, ...
2. **Label Leakage**: The target variable is known before it should be
3. **Train/Test Leakage**: Test data information leaks into training

### Why It's Dangerous

Leakage produces unrealistically good backtest results that cannot be replicated in live trading. Common symptoms:
- Sharpe ratios > 5 in backtests
- Near-perfect prediction accuracy
- Results that disappear in live trading

## How This Repository Prevents Leakage

### 1. Sequential Processing

All data processing is strictly sequential:

```python
for event in events:  # Chronological order
    features = compute_features(event)  # Only uses past data
    prediction = model.predict(features)
```

### 2. Timestamp Discipline

Every event has a timestamp. Features are timestamped at the moment they become available:

```python
# Feature at time t uses only data up to time t
feature_row = builder.on_event(event)  # Timestamp = event.timestamp
```

### 3. Safe Rolling Computations

Rolling features use `shift(1)` to exclude the current value:

```python
# WRONG: Uses current value (lookahead)
rolling_mean = df["price"].rolling(window=20).mean()

# CORRECT: Uses only past values
rolling_mean = df["price"].shift(1).rolling(window=20).mean()
```

### 4. Z-Score Without Lookahead

```python
def compute_rolling_zscore(values, window=20):
    # Mean and std computed on PAST values only
    mean = np.mean(recent[:-1])  # Exclude current
    std = np.std(recent[:-1], ddof=1)
    return (values[-1] - mean) / std
```

### 5. Label Shifting

When predicting future returns, labels must be shifted:

```python
# At time t, we want to predict return from t to t+1
future_return = (price[t+1] - price[t]) / price[t]

# For training: features at t, label = future_return
# But we must shift labels forward to align properly:
label = shift_labels(future_return, shift_periods=1)
```

### 6. Train/Test Time Separation

```python
# WRONG: Random split (destroys time structure)
train, test = sklearn.train_test_split(data)

# CORRECT: Time-based split
train = data[data.index <= split_date]
test = data[data.index > split_date + gap]
```

## Walk-Forward Implementation

Walk-forward validation is the gold standard for time series:

```
Period 1: [Train][Gap][Test]........................
Period 2: [    Train    ][Gap][Test]................
Period 3: [        Train        ][Gap][Test]........
```

### Usage

```python
from crypto_mm_research.evaluation.stability import walk_forward_split

for train_data, test_data in walk_forward_split(
    data,
    n_splits=5,
    min_train_size=timedelta(hours=1),
    test_size=timedelta(minutes=30),
    gap=timedelta(minutes=5),  # Prevents leakage
):
    model.fit(train_data)
    predictions = model.predict(test_data)
    evaluate(predictions)
```

### Key Parameters

- `min_train_size`: Minimum training data duration
- `test_size`: Duration of each test period
- `gap`: Time gap between train and test (prevents leakage from overlapping windows)

## Regime-Based Evaluation

Markets have different regimes (high/low volatility, trending/mean-reverting). Performance should be evaluated separately:

```python
from crypto_mm_research.evaluation.stability import (
    regime_split_by_volatility,
    evaluate_by_regime,
)

# Split by volatility
regimes = regime_split_by_volatility(data, n_regimes=3)

# Evaluate per regime
results = evaluate_by_regime(data, regimes, metric_fn)
# Returns: {'low_vol': {...}, 'medium_vol': {...}, 'high_vol': {...}}
```

### Why Regime Analysis Matters

A strategy might:
- Perform well in low volatility, fail in high volatility
- Work in trends, lose in ranges
- Have positive skew but occasional large losses

Regime analysis reveals these characteristics.

## Validation Checklist

Before trusting backtest results:

- [ ] Features use only data available at timestamp
- [ ] No future data in training
- [ ] Train/test splits have time gap
- [ ] Walk-forward validation performed
- [ ] Results stable across different time periods
- [ ] Performance evaluated per regime
- [ ] No target leakage in features (check correlations)

## Utilities

### Check for Target Leakage

```python
from crypto_mm_research.evaluation.leakage import check_for_target_leakage

leaky_features = check_for_target_leakage(features, target, threshold=0.99)
# Returns list of features with suspiciously high correlation
```

### Validate No Overlap

```python
from crypto_mm_research.evaluation.leakage import assert_no_overlap

assert_no_overlap(train_data, test_data, time_buffer=timedelta(minutes=1))
# Raises ValueError if overlap detected
```

### Stability Metrics

```python
from crypto_mm_research.evaluation.stability import compute_stability_metrics

stability = compute_stability_metrics(walk_forward_results, metric_name="sharpe_ratio")
# Returns: mean, std, min, max, coefficient of variation
```

## References

- Aronson, D. (2006). *Evidence-Based Technical Analysis*
- Bailey, D. et al. (2014). "Pseudo-Mathematics and Financial Charlatanism"
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*
