"""
Notebook 01: Order Book Features Demo
======================================

This notebook demonstrates:
1. Generating synthetic L2 order book data
2. Computing microstructure features
3. Visualizing feature distributions and relationships

Run with: python notebooks/01_orderbook_features_demo.py
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from crypto_mm_research.data.synthetic import SyntheticDataGenerator
from crypto_mm_research.features.builder import FeatureBuilder
from crypto_mm_research.features.microstructure import (
    compute_book_imbalance,
    compute_depth_imbalance,
    compute_microprice,
    compute_spread_bps,
)


def generate_data():
    """Generate synthetic market data."""
    print("Generating synthetic market data...")
    
    generator = SyntheticDataGenerator(
        symbol="BTC-USDT",
        start_price=50000.0,
        tick_size=0.1,
        volatility_annual=0.8,
        target_spread_bps=5.0,
        book_levels=10,
        random_seed=42,
    )
    
    # Generate 30 minutes of data
    events = generator.generate_to_list(
        duration_seconds=1800,
        events_per_second=10,
    )
    
    print(f"Generated {len(events)} events")
    return events


def compute_features(events):
    """Compute features from events."""
    print("Computing features...")
    
    builder = FeatureBuilder(
        symbol="BTC-USDT",
        window_seconds=20.0,
        max_history_seconds=300.0,
    )
    
    features_df = builder.process_events(events)
    print(f"Computed {len(features_df)} feature rows")
    
    return features_df


def plot_price_and_spread(features_df, output_path="outputs/01_price_spread.png"):
    """Plot price and spread over time."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Price
    axes[0].plot(features_df.index, features_df["mid_price"], label="Mid Price", color="blue")
    axes[0].plot(features_df.index, features_df["microprice"], label="Microprice", color="orange", alpha=0.7)
    axes[0].set_ylabel("Price ($)")
    axes[0].set_title("Price Over Time")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Spread
    axes[1].plot(features_df.index, features_df["spread_bps"], color="green")
    axes[1].set_ylabel("Spread (bps)")
    axes[1].set_title("Bid-Ask Spread")
    axes[1].grid(True, alpha=0.3)
    
    # Returns
    returns = features_df["returns_1s"] * 100  # Convert to %
    axes[2].plot(features_df.index, returns, color="purple", alpha=0.7)
    axes[2].axhline(y=0, color="black", linestyle="--", alpha=0.5)
    axes[2].set_ylabel("Return (%)")
    axes[2].set_xlabel("Time")
    axes[2].set_title("1-Second Returns")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_imbalance_features(features_df, output_path="outputs/01_imbalance.png"):
    """Plot imbalance features."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Book imbalance
    axes[0].plot(features_df.index, features_df["book_imbalance"], color="blue")
    axes[0].axhline(y=0, color="black", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Imbalance")
    axes[0].set_title("Top-of-Book Imbalance")
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].grid(True, alpha=0.3)
    
    # Depth imbalance
    axes[1].plot(features_df.index, features_df["depth_imbalance_5"], color="green")
    axes[1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
    axes[1].set_ylabel("Imbalance")
    axes[1].set_title("Depth-Weighted Imbalance (5 levels)")
    axes[1].set_ylim(-1.1, 1.1)
    axes[1].grid(True, alpha=0.3)
    
    # OFI
    ofi = features_df["ofi"]
    ofi_clipped = np.clip(ofi, -5, 5)  # Clip for visualization
    axes[2].plot(features_df.index, ofi_clipped, color="red", alpha=0.7)
    axes[2].axhline(y=0, color="black", linestyle="--", alpha=0.5)
    axes[2].set_ylabel("OFI")
    axes[2].set_xlabel("Time")
    axes[2].set_title("Order Flow Imbalance (clipped to [-5, 5])")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_volatility_and_zscore(features_df, output_path="outputs/01_volatility.png"):
    """Plot volatility and z-score features."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Realized volatility
    vol_bps = features_df["realized_vol_20"] * 10000  # Convert to bps
    axes[0].plot(features_df.index, vol_bps, color="purple")
    axes[0].set_ylabel("Vol (bps)")
    axes[0].set_title("Realized Volatility (20-period)")
    axes[0].grid(True, alpha=0.3)
    
    # Z-score
    axes[1].plot(features_df.index, features_df["zscore_mid_20"], color="orange")
    axes[1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
    axes[1].axhline(y=2, color="red", linestyle="--", alpha=0.3)
    axes[1].axhline(y=-2, color="red", linestyle="--", alpha=0.3)
    axes[1].set_ylabel("Z-Score")
    axes[1].set_title("Mid Price Z-Score (20-period)")
    axes[1].grid(True, alpha=0.3)
    
    # Trade intensity
    axes[2].plot(features_df.index, features_df["trade_intensity"], color="brown")
    axes[2].set_ylabel("Trades/sec")
    axes[2].set_xlabel("Time")
    axes[2].set_title("Trade Intensity")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_correlation_matrix(features_df, output_path="outputs/01_correlation.png"):
    """Plot feature correlation matrix."""
    # Select numeric features
    feature_cols = [
        "mid_price", "spread_bps", "book_imbalance",
        "depth_imbalance_5", "ofi", "trade_imbalance",
        "realized_vol_20", "returns_1s", "zscore_mid_20"
    ]
    
    corr = features_df[feature_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    
    # Labels
    ax.set_xticks(range(len(feature_cols)))
    ax.set_yticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=45, ha="right")
    ax.set_yticklabels(feature_cols)
    
    # Colorbar
    plt.colorbar(im, ax=ax)
    
    # Values
    for i in range(len(feature_cols)):
        for j in range(len(feature_cols)):
            text = ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    plt.close()


def print_feature_stats(features_df):
    """Print feature statistics."""
    print("\n" + "=" * 60)
    print("FEATURE STATISTICS")
    print("=" * 60)
    
    stats = features_df.describe()
    print(stats.to_string())
    
    print("\n" + "-" * 60)
    print("Key Observations:")
    print(f"- Average spread: {features_df['spread_bps'].mean():.2f} bps")
    print(f"- Spread range: {features_df['spread_bps'].min():.2f} - {features_df['spread_bps'].max():.2f} bps")
    print(f"- Avg book imbalance: {features_df['book_imbalance'].mean():.4f}")
    print(f"- Avg volatility: {features_df['realized_vol_20'].mean()*10000:.2f} bps")
    print(f"- Return autocorr(1): {features_df['returns_1s'].autocorr(lag=1):.4f}")
    print("=" * 60)


def main():
    """Main notebook execution."""
    print("=" * 60)
    print("NOTEBOOK 01: Order Book Features Demo")
    print("=" * 60)
    
    # Generate data
    events = generate_data()
    
    # Compute features
    features_df = compute_features(events)
    
    # Print statistics
    print_feature_stats(features_df)
    
    # Create plots
    print("\nGenerating plots...")
    plot_price_and_spread(features_df)
    plot_imbalance_features(features_df)
    plot_volatility_and_zscore(features_df)
    plot_correlation_matrix(features_df)
    
    # Save features to CSV
    output_csv = "outputs/01_features.csv"
    features_df.to_csv(output_csv)
    print(f"\nSaved features to {output_csv}")
    
    print("\n" + "=" * 60)
    print("Notebook 01 complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
