"""Report generator for experiment results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np


class ReportGenerator:
    """Generate markdown reports from experiment results."""
    
    def __init__(self, summary_df: pd.DataFrame, output_dir: Path) -> None:
        self.summary = summary_df
        self.output_dir = Path(output_dir)
    
    def generate(self) -> Path:
        """Generate full report."""
        report_path = self.output_dir / "report.md"
        
        with open(report_path, "w") as f:
            f.write(self._generate_header())
            f.write(self._generate_summary_stats())
            f.write(self._generate_top_runs())
            f.write(self._generate_parameter_sensitivity())
            f.write(self._generate_risk_analysis())
            f.write(self._generate_adverse_selection_analysis())
        
        return report_path
    
    def _generate_header(self) -> str:
        return f"""# Experiment Report

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Total Runs: {len(self.summary)}

"""
    
    def _generate_summary_stats(self) -> str:
        metrics = ["total_pnl", "sharpe_ratio", "max_drawdown_pct", "n_fills"]
        available = [m for m in metrics if m in self.summary.columns]
        
        if not available:
            return ""
        
        stats = self.summary[available].describe()
        
        md = "## Summary Statistics\n\n"
        md += "| Metric | Mean | Std | Min | 25% | 50% | 75% | Max |\n"
        md += "|--------|------|-----|-----|-----|-----|-----|-----|\n"
        
        for metric in available:
            row = stats.loc[:, metric]
            md += f"| {metric} | {row['mean']:.4f} | {row['std']:.4f} | "
            md += f"{row['min']:.4f} | {row['25%']:.4f} | {row['50%']:.4f} | "
            md += f"{row['75%']:.4f} | {row['max']:.4f} |\n"
        
        md += "\n"
        return md
    
    def _generate_top_runs(self) -> str:
        if "total_pnl" not in self.summary.columns:
            return ""
        
        top = self.summary.nlargest(10, "total_pnl")
        
        md = "## Top 10 Runs by Total PnL\n\n"
        md += "| Rank | Run | Total PnL | Sharpe | Max DD | Fills |\n"
        md += "|------|-----|-----------|--------|--------|-------|\n"
        
        for i, (idx, row) in enumerate(top.iterrows(), 1):
            run_name = row.get("run_name", f"run_{idx}")
            pnl = row.get("total_pnl", 0)
            sharpe = row.get("sharpe_ratio", 0)
            dd = row.get("max_drawdown_pct", 0)
            fills = row.get("n_fills", 0)
            
            md += f"| {i} | {run_name} | ${pnl:.2f} | {sharpe:.3f} | {dd:.2f}% | {fills} |\n"
        
        md += "\n"
        return md
    
    def _generate_parameter_sensitivity(self) -> str:
        param_cols = [c for c in self.summary.columns if c.startswith("param_")]
        
        if not param_cols or "total_pnl" not in self.summary.columns:
            return ""
        
        md = "## Parameter Sensitivity\n\n"
        
        for param in param_cols:
            param_name = param.replace("param_", "").replace("strategy.", "").replace("backtest.", "")
            
            grouped = self.summary.groupby(param)["total_pnl"].agg(["mean", "std", "count"])
            
            md += f"### {param_name}\n\n"
            md += "| Value | Avg PnL | Std PnL | Count |\n"
            md += "|-------|---------|---------|-------|\n"
            
            for value, row in grouped.iterrows():
                md += f"| {value} | ${row['mean']:.2f} | ${row['std']:.2f} | {row['count']} |\n"
            
            md += "\n"
        
        return md
    
    def _generate_risk_analysis(self) -> str:
        if "max_drawdown_pct" not in self.summary.columns:
            return ""
        
        md = "## Risk Analysis\n\n"
        
        dd = self.summary["max_drawdown_pct"]
        md += f"- **Average Max Drawdown**: {dd.mean():.2f}%\n"
        md += f"- **Worst Drawdown**: {dd.max():.2f}%\n"
        md += f"- **Best (Lowest) Drawdown**: {dd.min():.2f}%\n"
        md += f"- **Drawdown Std**: {dd.std():.2f}%\n\n"
        
        if "total_pnl" in self.summary.columns:
            corr = self.summary["total_pnl"].corr(self.summary["max_drawdown_pct"])
            md += f"- **PnL vs Drawdown Correlation**: {corr:.3f}\n\n"
        
        return md
    
    def _generate_adverse_selection_analysis(self) -> str:
        if "avg_spread_captured" not in self.summary.columns:
            return ""
        
        md = "## Adverse Selection Analysis\n\n"
        
        spread = self.summary["avg_spread_captured"]
        md += f"- **Average Spread Captured**: ${spread.mean():.4f}\n"
        md += f"- **Spread Std**: ${spread.std():.4f}\n"
        md += f"- **Min Spread**: ${spread.min():.4f}\n"
        md += f"- **Max Spread**: ${spread.max():.4f}\n\n"
        
        md += "### Spread Capture Distribution\n\n"
        md += "| Percentile | Value |\n"
        md += "|------------|-------|\n"
        
        for p in [10, 25, 50, 75, 90]:
            val = spread.quantile(p / 100)
            md += f"| {p}th | ${val:.4f} |\n"
        
        md += "\n"
        
        if "alignment_rate" in self.summary.columns:
            md += "## Alignment Quality\n\n"
            md += f"- **Average Alignment Rate**: {self.summary['alignment_rate'].mean():.1%}\n"
            md += f"- **Min Alignment Rate**: {self.summary['alignment_rate'].min():.1%}\n\n"
        
        return md
