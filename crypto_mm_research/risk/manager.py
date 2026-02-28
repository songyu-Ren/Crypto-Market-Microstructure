"""Risk manager for strategy-level risk controls."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum, auto


class RiskEventType(Enum):
    """Types of risk events."""
    POSITION_LIMIT_BREACH = auto()
    DRAWDOWN_BREACH = auto()
    VOLATILITY_SPIKE = auto()
    TRADE_RATE_EXCEEDED = auto()


@dataclass
class RiskEvent:
    """Record of a risk event."""
    
    timestamp: datetime
    event_type: RiskEventType
    message: str
    metrics: Dict[str, Any]
    action_taken: str


@dataclass
class RiskConfig:
    """Configuration for risk management."""
    
    max_position_absolute: float = 10.0
    max_position_notional: float = 500000.0
    
    max_drawdown_pct: float = 5.0
    drawdown_kill_switch: bool = True
    
    max_trades_per_minute: int = 100
    max_trades_per_hour: int = 1000
    
    max_volatility_annual: float = 2.0
    
    default_tif: str = "GTC"


class RiskManager:
    """Manage risk limits and trigger kill switches."""
    
    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self.events: List[RiskEvent] = []
        self._kill_switch_triggered: bool = False
        self._kill_switch_reason: Optional[str] = None
        self._kill_switch_time: Optional[datetime] = None
        
        self._trade_times: List[datetime] = []
        self._peak_equity: float = 0.0
    
    @property
    def kill_switch_active(self) -> bool:
        return self._kill_switch_triggered
    
    def check_position_limit(
        self,
        position_size: float,
        position_notional: float,
        timestamp: datetime,
    ) -> bool:
        if abs(position_size) > self.config.max_position_absolute:
            self._trigger_event(
                RiskEventType.POSITION_LIMIT_BREACH,
                f"Position size {position_size} exceeds limit {self.config.max_position_absolute}",
                {"position_size": position_size, "limit": self.config.max_position_absolute},
                "Block new orders in same direction",
                timestamp,
            )
            return False
        
        if position_notional > self.config.max_position_notional:
            self._trigger_event(
                RiskEventType.POSITION_LIMIT_BREACH,
                f"Position notional {position_notional} exceeds limit {self.config.max_position_notional}",
                {"position_notional": position_notional, "limit": self.config.max_position_notional},
                "Block new orders",
                timestamp,
            )
            return False
        
        return True
    
    def check_drawdown(
        self,
        current_equity: float,
        timestamp: datetime,
    ) -> bool:
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
        
        if self._peak_equity <= 0:
            return True
        
        drawdown_pct = (self._peak_equity - current_equity) / self._peak_equity * 100
        
        if drawdown_pct > self.config.max_drawdown_pct:
            if self.config.drawdown_kill_switch and not self._kill_switch_triggered:
                self._trigger_kill_switch(
                    f"Drawdown {drawdown_pct:.2f}% exceeds limit {self.config.max_drawdown_pct}%",
                    timestamp,
                )
                self._trigger_event(
                    RiskEventType.DRAWDOWN_BREACH,
                    f"Kill switch triggered due to drawdown",
                    {"drawdown_pct": drawdown_pct, "limit": self.config.max_drawdown_pct},
                    "Stop quoting, initiate emergency flat",
                    timestamp,
                )
            return False
        
        return True
    
    def check_trade_rate(
        self,
        timestamp: datetime,
    ) -> bool:
        self._trade_times.append(timestamp)
        
        cutoff_minute = timestamp - timedelta(minutes=1)
        cutoff_hour = timestamp - timedelta(hours=1)
        
        self._trade_times = [t for t in self._trade_times if t > cutoff_hour]
        
        trades_last_minute = sum(1 for t in self._trade_times if t > cutoff_minute)
        trades_last_hour = len(self._trade_times)
        
        if trades_last_minute > self.config.max_trades_per_minute:
            self._trigger_event(
                RiskEventType.TRADE_RATE_EXCEEDED,
                f"Trade rate {trades_last_minute}/min exceeds limit",
                {"trades_per_minute": trades_last_minute, "limit": self.config.max_trades_per_minute},
                "Throttle order submission",
                timestamp,
            )
            return False
        
        if trades_last_hour > self.config.max_trades_per_hour:
            self._trigger_event(
                RiskEventType.TRADE_RATE_EXCEEDED,
                f"Trade rate {trades_last_hour}/hour exceeds limit",
                {"trades_per_hour": trades_last_hour, "limit": self.config.max_trades_per_hour},
                "Throttle order submission",
                timestamp,
            )
            return False
        
        return True
    
    def _trigger_event(
        self,
        event_type: RiskEventType,
        message: str,
        metrics: Dict[str, Any],
        action: str,
        timestamp: datetime,
    ) -> None:
        event = RiskEvent(
            timestamp=timestamp,
            event_type=event_type,
            message=message,
            metrics=metrics,
            action_taken=action,
        )
        self.events.append(event)
    
    def _trigger_kill_switch(self, reason: str, timestamp: datetime) -> None:
        self._kill_switch_triggered = True
        self._kill_switch_reason = reason
        self._kill_switch_time = timestamp
    
    def reset_kill_switch(self) -> None:
        self._kill_switch_triggered = False
        self._kill_switch_reason = None
        self._kill_switch_time = None
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "kill_switch_active": self._kill_switch_triggered,
            "kill_switch_reason": self._kill_switch_reason,
            "kill_switch_time": self._kill_switch_time,
            "total_events": len(self.events),
            "events_by_type": {
                event_type.name: sum(1 for e in self.events if e.event_type == event_type)
                for event_type in RiskEventType
            },
        }
