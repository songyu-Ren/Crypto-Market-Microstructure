"""Latency model with jitter and out-of-order handling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional
import numpy as np


@dataclass(frozen=True)
class TimedEvent:
    """Event with both event time and arrival time."""
    
    event: object
    event_ts: datetime
    arrival_ts: datetime
    sequence: int
    
    def __lt__(self, other: "TimedEvent") -> bool:
        if self.arrival_ts != other.arrival_ts:
            return self.arrival_ts < other.arrival_ts
        return self.sequence < other.sequence


class LatencyModel:
    """Models network latency with jitter."""
    
    def __init__(
        self,
        base_latency_ms: float = 10.0,
        jitter_ms: float = 5.0,
        jitter_type: str = "uniform",
        random_seed: int = 42,
    ) -> None:
        self.base_latency = timedelta(milliseconds=base_latency_ms)
        self.jitter_ms = jitter_ms
        self.jitter_type = jitter_type
        self.rng = np.random.default_rng(random_seed)
    
    def get_latency(self) -> timedelta:
        if self.jitter_type == "fixed":
            jitter = timedelta(milliseconds=0)
        elif self.jitter_type == "uniform":
            jitter_ms = self.rng.uniform(-self.jitter_ms, self.jitter_ms)
            jitter = timedelta(milliseconds=float(jitter_ms))
        elif self.jitter_type == "normal":
            jitter_ms = self.rng.normal(0, self.jitter_ms / 2)
            jitter = timedelta(milliseconds=float(jitter_ms))
        else:
            jitter = timedelta(milliseconds=0)
        
        return self.base_latency + jitter
    
    def compute_arrival_time(self, event_ts: datetime) -> datetime:
        return event_ts + self.get_latency()


class ArrivalTimeGenerator:
    """Generates events with arrival times."""
    
    def __init__(
        self,
        market_data_latency: LatencyModel,
        order_latency: LatencyModel,
        reordering_window_ms: float = 50.0,
    ) -> None:
        self.market_data_latency = market_data_latency
        self.order_latency = order_latency
        self.reordering_window = timedelta(milliseconds=reordering_window_ms)
        
        self._buffer: List[TimedEvent] = []
        self._sequence = 0
        self._last_emit_ts: Optional[datetime] = None
    
    def add_market_event(self, event: object, event_ts: datetime) -> List[TimedEvent]:
        latency = self.market_data_latency.get_latency()
        arrival_ts = event_ts + latency
        
        timed = TimedEvent(
            event=event,
            event_ts=event_ts,
            arrival_ts=arrival_ts,
            sequence=self._sequence,
        )
        self._sequence += 1
        
        self._buffer.append(timed)
        self._buffer.sort()
        
        return self._emit_ready_events(arrival_ts)
    
    def _emit_ready_events(self, current_arrival_ts: datetime) -> List[TimedEvent]:
        ready = []
        cutoff = current_arrival_ts - self.reordering_window
        
        while self._buffer and self._buffer[0].arrival_ts <= cutoff:
            ready.append(self._buffer.pop(0))
        
        if ready:
            self._last_emit_ts = ready[-1].arrival_ts
        
        return ready
    
    def flush(self) -> List[TimedEvent]:
        ready = self._buffer.copy()
        self._buffer.clear()
        return ready


class SimpleReorderBuffer:
    """Simple buffer for handling out-of-order events."""
    
    def __init__(
        self,
        max_delay_ms: float = 100.0,
        warn_on_reorder: bool = True,
    ) -> None:
        self.max_delay = timedelta(milliseconds=max_delay_ms)
        self.warn_on_reorder = warn_on_reorder
        
        self._buffer: List[TimedEvent] = []
        self._last_event_ts: Optional[datetime] = None
        self._reorder_count = 0
    
    def add(self, timed_event: TimedEvent) -> List[TimedEvent]:
        if self._last_event_ts and timed_event.event_ts < self._last_event_ts:
            self._reorder_count += 1
            if self.warn_on_reorder:
                print(f"WARNING: Out-of-order event detected.")
        
        self._buffer.append(timed_event)
        self._buffer.sort()
        
        if self._last_event_ts is None or timed_event.event_ts > self._last_event_ts:
            self._last_event_ts = timed_event.event_ts
        
        ready = []
        cutoff = timed_event.arrival_ts - self.max_delay
        
        while self._buffer and self._buffer[0].arrival_ts <= cutoff:
            ready.append(self._buffer.pop(0))
        
        return ready
    
    def flush(self) -> List[TimedEvent]:
        ready = self._buffer.copy()
        self._buffer.clear()
        return ready
    
    @property
    def stats(self) -> dict:
        return {
            "reorder_count": self._reorder_count,
            "buffer_size": len(self._buffer),
        }
