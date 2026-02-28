"""Label construction for future returns and directions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from enum import Enum

import numpy as np
import pandas as pd


class DirectionLabel(Enum):
    """Direction labels for classification."""
    DOWN = -1
    FLAT = 0
    UP = 1


@dataclass
class LabelConfig:
    """Configuration for label construction."""
    
    horizons: List[int]
    direction_threshold_bps: float = 5.0
    use_log_returns: bool = True
    clip_outliers: Optional[float] = None


class LabelConstructor:
    """Construct labels for microstructure research with strict no-lookahead."""
    
    def __init__(self, config: LabelConfig) -> None:
        self.config = config
    
    def construct_labels(
        self,
        mid_prices: pd.Series,
        event_timestamps: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Construct labels for given mid price series."""
        labels_dict = {}
        
        for h in self.config.horizons:
            horizon_td = timedelta(seconds=h)
            
            returns = self._compute_future_returns(
                mid_prices, event_timestamps, horizon_td
            )
            
            directions = self._compute_direction_labels(returns)
            
            labels_dict[f"return_{h}s"] = returns
            labels_dict[f"direction_{h}s"] = directions
            labels_dict[f"mid_change_{h}s"] = self._compute_future_mid_changes(
                mid_prices, event_timestamps, horizon_td
            )
        
        labels_df = pd.DataFrame(labels_dict, index=event_timestamps)
        labels_df = labels_df.shift(-1)
        
        return labels_df
    
    def _compute_future_returns(
        self,
        mid_prices: pd.Series,
        timestamps: pd.DatetimeIndex,
        horizon: timedelta,
    ) -> pd.Series:
        """Compute future returns at each timestamp."""
        returns = pd.Series(index=timestamps, dtype=float)
        
        for i, ts in enumerate(timestamps):
            current_price = mid_prices.iloc[i]
            future_ts = ts + horizon
            
            future_mask = timestamps > ts
            if not future_mask.any():
                returns.iloc[i] = np.nan
                continue
            
            future_indices = timestamps[future_mask]
            closest_future_idx = future_indices[future_indices <= future_ts]
            
            if len(closest_future_idx) == 0:
                future_price = mid_prices.iloc[i + 1] if i + 1 < len(mid_prices) else np.nan
            else:
                future_price = mid_prices.loc[closest_future_idx[-1]]
            
            if pd.isna(future_price) or current_price == 0:
                returns.iloc[i] = np.nan
            else:
                if self.config.use_log_returns:
                    returns.iloc[i] = np.log(future_price / current_price)
                else:
                    returns.iloc[i] = (future_price - current_price) / current_price
        
        return returns
    
    def _compute_future_mid_changes(
        self,
        mid_prices: pd.Series,
        timestamps: pd.DatetimeIndex,
        horizon: timedelta,
    ) -> pd.Series:
        """Compute absolute mid changes."""
        changes = pd.Series(index=timestamps, dtype=float)
        
        for i, ts in enumerate(timestamps):
            current_price = mid_prices.iloc[i]
            future_ts = ts + horizon
            
            future_mask = timestamps > ts
            if not future_mask.any():
                changes.iloc[i] = np.nan
                continue
            
            future_indices = timestamps[future_mask]
            closest_future_idx = future_indices[future_indices <= future_ts]
            
            if len(closest_future_idx) == 0:
                future_price = mid_prices.iloc[i + 1] if i + 1 < len(mid_prices) else np.nan
            else:
                future_price = mid_prices.loc[closest_future_idx[-1]]
            
            changes.iloc[i] = future_price - current_price
        
        return changes
    
    def _compute_direction_labels(self, returns: pd.Series) -> pd.Series:
        """Convert returns to direction labels."""
        threshold = self.config.direction_threshold_bps / 10000.0
        
        directions = pd.Series(index=returns.index, dtype=int)
        
        directions[returns > threshold] = DirectionLabel.UP.value
        directions[returns < -threshold] = DirectionLabel.DOWN.value
        directions[(returns >= -threshold) & (returns <= threshold)] = DirectionLabel.FLAT.value
        
        return directions
    
    def validate_no_lookahead(self, labels_df: pd.DataFrame) -> bool:
        """Validate that labels don't contain future information."""
        for col in labels_df.columns:
            if "return" in col or "direction" in col:
                if not labels_df[col].iloc[-3:].isna().all():
                    print(f"WARNING: Potential lookahead in {col}")
                    return False
        
        return True


def merge_features_labels(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    tolerance: timedelta = timedelta(seconds=1),
) -> pd.DataFrame:
    """Merge features and labels with time alignment."""
    features_reset = features.reset_index()
    labels_reset = labels.reset_index()
    
    merged = pd.merge_asof(
        features_reset.sort_values("timestamp"),
        labels_reset.sort_values("timestamp"),
        on="timestamp",
        tolerance=tolerance,
        direction="forward",
    )
    
    return merged.set_index("timestamp")
