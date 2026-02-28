"""Integration tests for latency and out-of-order handling."""

import pytest
from datetime import datetime, timedelta

from crypto_mm_research.execution.latency import (
    LatencyModel,
    ArrivalTimeGenerator,
    SimpleReorderBuffer,
    TimedEvent,
)


class TestLatencyIntegration:
    """Integration tests for latency model."""
    
    def test_end_to_end_latency_simulation(self):
        ""Test full latency simulation pipeline."""
        # Create latency models
        md_latency = LatencyModel(
            base_latency_ms=10.0,
            jitter_ms=2.0,
            jitter_type="uniform",
            random_seed=42,
        )
        
        order_latency = LatencyModel(
            base_latency_ms=20.0,
            jitter_ms=5.0,
            jitter_type="uniform",
            random_seed=43,
        )
        
        # Create arrival time generator
        generator = ArrivalTimeGenerator(
            market_data_latency=md_latency,
            order_latency=order_latency,
            reordering_window_ms=30.0,
        )
        
        # Simulate market events
        base_ts = datetime(2024, 1, 1, 12, 0, 0)
        
        class MockEvent:
            def __init__(self, name):
                self.name = name
        
        # Add events at different times
        events = []
        for i in range(10):
            event_ts = base_ts + timedelta(milliseconds=i * 10)
            ready = generator.add_market_event(MockEvent(f"event_{i}"), event_ts)
            events.extend(ready)
        
        # Flush remaining
        remaining = generator.flush()
        events.extend(remaining)
        
        # Verify all events processed
        assert len(events) == 10
        
        # Verify chronological by arrival time
        for i in range(1, len(events)):
            assert events[i].arrival_ts >= events[i-1].arrival_ts
    
    def test_out_of_order_handling(self):
        ""Test out-of-order event handling."""
        buffer = SimpleReorderBuffer(
            max_delay_ms=50.0,
            warn_on_reorder=False,
        )
        
        base_ts = datetime(2024, 1, 1, 12, 0, 0)
        
        class MockEvent:
            pass
        
        # Add events out of order
        events_data = [
            (base_ts + timedelta(ms=20), base_ts + timedelta(ms=30)),  # Event 2, arrives 3rd
            (base_ts + timedelta(ms=10), base_ts + timedelta(ms=25)),  # Event 1, arrives 2nd
            (base_ts + timedelta(ms=30), base_ts + timedelta(ms=35)),  # Event 3, arrives last
        ]
        
        all_ready = []
        for event_ts, arrival_ts in events_data:
            timed = TimedEvent(
                event=MockEvent(),
                event_ts=event_ts,
                arrival_ts=arrival_ts,
                sequence=0,
            )
            ready = buffer.add(timed)
            all_ready.extend(ready)
        
        # Flush remaining
        remaining = buffer.flush()
        all_ready.extend(remaining)
        
        # All events should be processed
        assert len(all_ready) == 3
        
        # Check reorder detection
        assert buffer.stats["reorder_count"] > 0
    
    def test_reordering_window_boundary(self):
        ""Test events at reordering window boundary."""
        buffer = SimpleReorderBuffer(max_delay_ms=100.0)
        
        base_ts = datetime(2024, 1, 1, 12, 0, 0)
        
        class MockEvent:
            pass
        
        # Event that arrives "early" but within window
        early_event = TimedEvent(
            event=MockEvent(),
            event_ts=base_ts + timedelta(ms=50),
            arrival_ts=base_ts + timedelta(ms=100),
            sequence=0,
        )
        
        # Event that arrives "late" to flush
        late_event = TimedEvent(
            event=MockEvent(),
            event_ts=base_ts + timedelta(ms=200),
            arrival_ts=base_ts + timedelta(ms=250),
            sequence=1,
        )
        
        ready1 = buffer.add(early_event)
        assert len(ready1) == 0  # Not ready yet
        
        ready2 = buffer.add(late_event)
        assert len(ready2) == 1  # First event now ready
        assert ready2[0].event_ts == base_ts + timedelta(ms=50)
