"""CLI entry point for running experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from crypto_mm_research.experiments.runner import ExperimentRunner, ExperimentConfig
from crypto_mm_research.experiments.report import ReportGenerator
import yaml


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run batch experiments with parameter grid"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip report generation",
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    
    exp_config = ExperimentConfig(
        name=config_dict.get("name", "experiment"),
        base_config=config_dict.get("base_config", {}),
        parameter_grid=config_dict.get("parameter_grid", {}),
        random_seed=config_dict.get("random_seed", 42),
    )
    
    # Run experiments
    runner = ExperimentRunner(exp_config, output_dir=args.output_dir)
    summary = runner.run_all(verbose=True)
    
    # Generate report
    if not args.no_report:
        report_gen = ReportGenerator(summary, runner.output_dir)
        report_path = report_gen.generate()
        print(f"\nReport generated: {report_path}")
    
    # Print top runs
    print("\n" + "=" * 60)
    print("TOP 5 RUNS BY TOTAL PnL")
    print("=" * 60)
    top = runner.get_top_runs(n=5, metric="total_pnl")
    print(top[["run_name", "total_pnl", "sharpe_ratio", "max_drawdown_pct"]].to_string())
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
