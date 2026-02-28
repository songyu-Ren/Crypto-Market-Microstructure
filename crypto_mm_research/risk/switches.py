"""Kill switch implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional


class KillSwitch(ABC):
    """Abstract base class for kill switches."""
    
    def __init__(self, name: str) -> None:
        self.name = name
        self._triggered: bool = False
        self._trigger_time: Optional[datetime] = None
        self._reason: Optional[str] = None
    
    @property
    def triggered(self) -> bool:
        return self._triggered
    
    @property
    def trigger_time(self) -> Optional[datetime]:
        return self._trigger_time
    
    @property
    def reason(self) -> Optional[str]:
        return self._reason
    
    def trigger(self, reason: str, timestamp: Optional[datetime] = None) -> None:
        self._triggered = True
        self._trigger_time = timestamp or datetime.now()
        self._reason = reason
    
    def reset(self) -> None:
        self._triggered = False
        self._trigger_time = None
        self._reason = None
    
    @abstractmethod
    def check(self, **kwargs) -> bool:
        pass


class DrawdownKillSwitch(KillSwitch):
    """Kill switch based on drawdown."""
    
    def __init__(
        self,
        max_drawdown_pct: float = 5.0,
        lookback_periods: int = 1,
    ) -> None:
        super().__init__("DrawdownKillSwitch")
        self.max_drawdown_pct = max_drawdown_pct
        self.lookback_periods = lookback_periods
        self._peak_equity: float = 0.0
        self._equity_history: list[float] = []
    
    def check(
        self,
        current_equity: float,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        if self._triggered:
            return False
        
        self._equity_history.append(current_equity)
        if len(self._equity_history) > self.lookback_periods:
            self._equity_history.pop(0)
        
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
        
        if self._peak_equity <= 0:
            return True
        
        drawdown_pct = (self._peak_equity - current_equity) / self._peak_equity * 100
        
        if drawdown_pct > self.max_drawdown_pct:
            self.trigger(
                f"Drawdown {drawdown_pct:.2f}% exceeds limit {self.max_drawdown_pct}%",
                timestamp,
            )
            return False
        
        return True
    
    def reset(self) -> None:
        super().reset()
        self._peak_equity = 0.0
        self._equity_history.clear()
