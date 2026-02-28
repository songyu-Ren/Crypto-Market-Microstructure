# Metrics Definitions

This document provides precise definitions for all metrics computed in the backtest.

## PnL Metrics

### Total PnL
```
Total PnL = Final Equity - Initial Cash
```
The absolute profit/loss in currency terms.

### Realized PnL
```
Realized PnL = Σ(closed_position_pnl) - Σ(fees)
```
Profit/loss from closed trades, net of fees.

For a round-trip (buy then sell):
```
Realized PnL = (sell_price - buy_price) × size - fees
```

### Unrealized PnL
```
Unrealized PnL = Σ(position_size × (mark_price - avg_entry_price))
```
Mark-to-market value of open positions.

### Total Return %
```
Total Return % = 100 × (Final Equity / Initial Cash - 1)
```
Percentage return on initial capital.

## Trade Metrics

### Number of Fills
Total number of executed orders (both buy and sell).

### Number of Buys / Sells
```
N_Buys = count(fills where side = BUY)
N_Sells = count(fills where side = SELL)
```

### Average Trade Size
```
Avg Trade Size = Total Volume / Number of Fills
```

### Total Volume
```
Total Volume = Σ(fill_size)
```
Sum of all fill sizes in base currency units.

## Spread Capture Metrics

### Average Spread Captured
```
For each round-trip (buy then sell or vice versa):
    Spread Captured = |sell_price - buy_price|

Avg Spread Captured = mean(Spread Captured across all round-trips)
```

### Spread PnL Estimate
```
Spread PnL Estimate = Avg Spread Captured × Number of Round-trips / 2
```
Approximate PnL from spread capture (excludes adverse selection).

## Risk Metrics

### Sharpe Ratio
```
Sharpe Ratio = √N × (mean(returns) - risk_free_rate) / std(returns)
```
Where:
- `returns`: Period-over-period equity returns
- `N`: Number of periods per year (estimated from data frequency)
- `risk_free_rate`: Annual risk-free rate (default: 0)

**Caution**: Sharpe ratio assumes normally distributed returns. For high-frequency strategies, consider:
- Sortino ratio (downside deviation only)
- Calmar ratio (return / max drawdown)

### Maximum Drawdown %
```
Peak_t = max(equity[0:t])
Drawdown_t = (equity_t - Peak_t) / Peak_t
Max Drawdown % = 100 × min(Drawdown_t)
```
The largest peak-to-trough decline.

### Maximum Drawdown Duration
```
Duration = max(time_in_drawdown)
```
Longest continuous period spent in drawdown (in days).

### Volatility (Annualized)
```
Volatility = std(returns) × √N
```
Standard deviation of returns, annualized.

## Inventory Metrics

### Average Inventory
```
Avg Inventory = mean(position_size over time)
```
Time-weighted average position.

### Maximum Inventory
```
Max Inventory = max(|position_size|)
```
Largest absolute position reached.

### Inventory Standard Deviation
```
Inventory Std = std(position_size)
```
Volatility of position size.

## Fee Metrics

### Total Fees
```
Total Fees = Σ(fill_notional × fee_rate)
```
Sum of all fees paid.

### Fee / PnL Ratio
```
Fee/PnL Ratio = Total Fees / |Total PnL|
```
Ratio of fees to gross PnL. Values > 1 indicate fees exceed gross profits.

## Turnover

```
Turnover = Total Volume / Initial Cash
```
Ratio of trading volume to capital. Higher turnover means:
- More trading activity
- Higher fee burden
- Potentially more market impact

## Market Making Specific Metrics

### Adverse Selection Estimate
```
Adverse Selection = Spread PnL Estimate - Realized PnL
```
Rough estimate of losses to informed traders. Positive means the strategy captured less than the quoted spread.

### Inventory Turnover
```
Inventory Turnover = Total Volume / (2 × Avg Absolute Inventory)
```
How often the position "turns over" completely.

### Time in Market
```
Time in Market % = 100 × (time_with_nonzero_position / total_time)
```
Percentage of time with open position.

## Implementation Details

### Equity Curve Sampling

The equity curve is sampled at regular intervals (default: 1 second) to:
1. Reduce memory usage
2. Provide consistent time series for analysis
3. Enable drawdown computation

### Return Calculation

```python
returns = equity.pct_change().dropna()
```
Simple percentage returns, not log returns.

### Annualization

For high-frequency data, annualization assumes:
```
Periods per year ≈ 365 × 24 × 3600 / avg_period_seconds
```

This is approximate; actual trading days/hours may differ.

## Interpretation Guidelines

### Sharpe Ratio
- < 1: Poor risk-adjusted returns
- 1-2: Acceptable
- 2-3: Good
- > 3: Excellent (but verify no leakage!)

### Max Drawdown
- < 5%: Conservative
- 5-10%: Moderate
- 10-20%: Aggressive
- > 20%: Very risky

### Fee/PnL Ratio
- < 0.1: Fees negligible
- 0.1-0.3: Reasonable
- 0.3-0.5: High
- > 0.5: Fees dominate

### Inventory
- Near 0: Well-balanced market maker
- Large positive: Long-biased
- Large negative: Short-biased
- High variance: Poor inventory control

## Limitations

1. **Sharpe Ratio**: Assumes normal returns; misleading for strategies with tail risk
2. **Max Drawdown**: Historical measure; future drawdowns may be larger
3. **Spread Capture**: Estimate only; doesn't account for partial fills
4. **Volatility**: Past volatility may not predict future volatility

Always use multiple metrics and perform out-of-sample testing.
