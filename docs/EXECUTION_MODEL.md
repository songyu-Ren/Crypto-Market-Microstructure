# Execution Model Documentation

## Overview

The execution model simulates how orders interact with the market, including:
- Queue position and priority
- Partial fills
- Trade-through logic
- Latency and jitter
- Event time vs arrival time

## Execution Models

### NaiveExecutionModel

Simplified fill logic:
- Bid fills if `bid >= best_ask` (crossed market)
- Ask fills if `ask <= best_bid` (crossed market)
- Trade-through fills on aggressive trades
- Full fills only (no partials)

Use case: Quick backtests, strategy development

### QueueExecutionModel

Realistic queue-based execution:
- Queue position estimation at submission
- Queue ahead tracking
- Partial fills as queue is consumed
- Trade-through with queue consumption

Use case: Realistic market making simulation

## Queue Mechanics

### Queue Position Estimation

When submitting at best bid/ask:
```
queue_ahead = visible_size * queue_ahead_ratio
```

Typical `queue_ahead_ratio` values:
- 0.0: At front of queue (optimistic)
- 0.5: Middle of queue (realistic)
- 1.0: Back of queue (pessimistic)

### Queue Consumption

When a trade hits the level:
1. First consumes `queue_ahead`
2. Then fills our order (partially or fully)
3. Remaining trade volume goes behind us

## Latency Model

### Event Time vs Arrival Time

```
event_ts: When event occurred in the market
arrival_ts: When event reaches our system
arrival_ts = event_ts + base_latency + jitter
```

### Jitter Types

- `fixed`: No jitter, deterministic latency
- `uniform`: Uniform random jitter in [-jitter, +jitter]
- `normal`: Normal distribution jitter

### Out-of-Order Handling

The `ArrivalTimeGenerator` handles out-of-order events:
1. Events buffered by arrival time
2. Reordering window waits for late events
3. Events outside window are emitted with warning

```python
generator = ArrivalTimeGenerator(
    market_data_latency=LatencyModel(base_latency_ms=10),
    order_latency=LatencyModel(base_latency_ms=20),
    reordering_window_ms=50,  # Wait up to 50ms
)
```

## Order Lifecycle

### States

```
PENDING_SUBMIT -> ACTIVE -> PARTIALLY_FILLED -> FILLED
                    |
                    -> PENDING_CANCEL -> CANCELED
                    |
                    -> PENDING_REPLACE -> (new order)
```

### Time in Force

- `GTC`: Good till canceled (default)
- `IOC`: Immediate or cancel
- `FOK`: Fill or kill (not fully implemented)

## Fill Metadata

Each fill includes:
- `fill_id`: Unique identifier
- `order_id`: Reference to order
- `fill_ts`: When fill occurred (event time)
- `arrival_ts`: When fill reported
- `mid_at_fill`: Mid price at fill time
- `spread_at_fill`: Spread at fill time
- `queue_ahead_before/after`: Queue position (if applicable)

## Configuration

### Queue Model Parameters

```yaml
execution:
  model: "queue"
  tick_size: 0.1
  lot_size: 0.001
  maker_fee_rate: 0.0002
  taker_fee_rate: 0.0005
  queue_ahead_ratio: 0.5
  allow_partial_fills: true
  trade_through_enabled: true
```

### Latency Parameters

```yaml
latency:
  market_data:
    base_ms: 10
    jitter_ms: 5
    type: "uniform"
  orders:
    base_ms: 20
    jitter_ms: 10
    type: "uniform"
  reordering_window_ms: 50
```

## Best Practices

1. **Use Queue Model for MM Research**: The naive model misses queue position effects critical for market making.

2. **Calibrate Queue Ratio**: Use 0.5 as baseline, test sensitivity with 0.3-0.7 range.

3. **Include Latency**: Even 10-20ms latency changes fill rates significantly.

4. **Monitor Alignment**: Always check `alignment_rate` in metrics.

5. **Validate with Real Data**: Compare simulated vs real fill rates when possible.
