"""Alignment tools for fills and adverse selection analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class AlignmentConfig:
    """Configuration for alignment operations."""
    
    max_time_tolerance: timedelta = timedelta(seconds=5)
    adverse_selection_horizon: timedelta = timedelta(seconds=10)
    mark_missing_if_exceeded: bool = True


class AlignmentTool:
    """Align fills to market data for analysis."""
    
    def __init__(self, config: AlignmentConfig) -> None:
        self.config = config
    
    def align_fills_to_mid(
        self,
        fills: pd.DataFrame,
        mid_prices: pd.Series,
    ) -> pd.DataFrame:
        """Align fill timestamps to nearest mid prices."""
        fills_aligned = fills.copy()
        fills_aligned["mid_at_fill"] = np.nan
        fills_aligned["mid_alignment_error_ms"] = np.nan
        fills_aligned["alignment_status"] = "pending"
        
        for fill_ts in fills.index:
            closest_idx, error_ms = self._find_closest_timestamp(
                fill_ts, mid_prices.index
            )
            
            if error_ms is not None and error_ms <= self.config.max_time_tolerance.total_seconds() * 1000:
                fills_aligned.loc[fill_ts, "mid_at_fill"] = mid_prices.iloc[closest_idx]
                fills_aligned.loc[fill_ts, "mid_alignment_error_ms"] = error_ms
                fills_aligned.loc[fill_ts, "alignment_status"] = "ok"
            else:
                fills_aligned.loc[fill_ts, "alignment_status"] = "fill_missing"
                if self.config.mark_missing_if_exceeded:
                    fills_aligned.loc[fill_ts, "mid_alignment_error_ms"] = error_ms
        
        return fills_aligned
    
    def compute_adverse_selection(
        self,
        fills_aligned: pd.DataFrame,
        mid_prices: pd.Series,
    ) -> pd.DataFrame:
        """Compute adverse selection by aligning to future mid prices."""
        result = fills_aligned.copy()
        result["mid_at_horizon"] = np.nan
        result["horizon_alignment_error_ms"] = np.nan
        result["adverse_selection"] = np.nan
        
        for fill_ts in result.index:
            if result.loc[fill_ts, "alignment_status"] == "fill_missing":
                result.loc[fill_ts, "alignment_status"] = "both_missing"
                continue
            
            horizon_ts = fill_ts + self.config.adverse_selection_horizon
            
            closest_idx, error_ms = self._find_closest_timestamp(
                horizon_ts, mid_prices.index
            )
            
            if error_ms is not None and error_ms <= self.config.max_time_tolerance.total_seconds() * 1000:
                mid_at_horizon = mid_prices.iloc[closest_idx]
                mid_at_fill = result.loc[fill_ts, "mid_at_fill"]
                
                result.loc[fill_ts, "mid_at_horizon"] = mid_at_horizon
                result.loc[fill_ts, "horizon_alignment_error_ms"] = error_ms
                result.loc[fill_ts, "adverse_selection"] = mid_at_horizon - mid_at_fill
                
                if result.loc[fill_ts, "is_maker"]:
                    side = result.loc[fill_ts, "side"]
                    fill_price = result.loc[fill_ts, "price"]
                    
                    if side == "buy":
                        result.loc[fill_ts, "realized_half_spread"] = mid_at_fill - fill_price
                    else:
                        result.loc[fill_ts, "realized_half_spread"] = fill_price - mid_at_fill
            else:
                if result.loc[fill_ts, "alignment_status"] == "ok":
                    result.loc[fill_ts, "alignment_status"] = "horizon_missing"
                else:
                    result.loc[fill_ts, "alignment_status"] = "both_missing"
                
                if self.config.mark_missing_if_exceeded:
                    result.loc[fill_ts, "horizon_alignment_error_ms"] = error_ms
        
        return result
    
    def _find_closest_timestamp(
        self,
        target: datetime,
        candidates: pd.DatetimeIndex,
    ) -> Tuple[int, Optional[float]]:
        """Find closest timestamp in candidates to target."""
        if len(candidates) == 0:
            return 0, None
        
        idx = candidates.searchsorted(target)
        
        if idx == 0:
            closest_idx = 0
        elif idx >= len(candidates):
            closest_idx = len(candidates) - 1
        else:
            before_diff = abs((target - candidates[idx - 1]).total_seconds())
            after_diff = abs((candidates[idx] - target).total_seconds())
            
            if before_diff < after_diff:
                closest_idx = idx - 1
            else:
                closest_idx = idx
        
        error_ms = abs((target - candidates[closest_idx]).total_seconds()) * 1000
        
        return closest_idx, error_ms
    
    def compute_alignment_stats(self, aligned_df: pd.DataFrame) -> Dict:
        """Compute statistics about alignment quality."""
        total = len(aligned_df)
        
        status_counts = aligned_df["alignment_status"].value_counts().to_dict()
        
        stats = {
            "total_fills": total,
            "fully_aligned": status_counts.get("ok", 0),
            "fill_missing": status_counts.get("fill_missing", 0),
            "horizon_missing": status_counts.get("horizon_missing", 0),
            "both_missing": status_counts.get("both_missing", 0),
            "alignment_rate": status_counts.get("ok", 0) / total if total > 0 else 0,
        }
        
        ok_mask = aligned_df["alignment_status"] == "ok"
        if ok_mask.any():
            stats["avg_mid_alignment_error_ms"] = aligned_df.loc[ok_mask, "mid_alignment_error_ms"].mean()
            stats["avg_horizon_alignment_error_ms"] = aligned_df.loc[ok_mask, "horizon_alignment_error_ms"].mean()
        
        return stats
