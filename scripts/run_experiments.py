"""
Run the full multi-strategy comparison experiment.

Generates scenarios, runs all configured agent strategies across
N episodes each, evaluates results, and saves outputs including
summary CSV, figures, and a strategy leaderboard.

Usage
-----
    python scripts/run_experiments.py [--config configs/default.yaml]
                                      [--strategies baseline rule_augmented ...]
                                      [--n-episodes 20]
                                      [--output-dir results]

Outputs
-------
    results/tables/experiment_summary.csv
    results/figures/strategy_comparison.png
    results/figures/metric_distributions.png
    results/figures/score_heatmap.png
    results/sample_runs/experiment_<id>.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from norm_dialogue_framework.config import load_config
from norm_dialogue_framework.experiments.compare_strategies import StrategyComparison
from norm_dialogue_framework.utils import ensure_dir, get_logger, set_seed
from norm_dialogue_framework.visualization.plots import (
    plot_metric_distributions,
    plot_radar_chart,
    plot_score_heatmap,
    plot_strategy_comparison,
)

logger = get_logger("run_experiments")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-strategy comparison experiment.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Strategies to compare (default: all configured).",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=None,
        help="Episodes per strategy (default: from config).",
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.project.random_seed
    set_seed(seed)

    output_dir = Path(args.output_dir)
    figures_dir = ensure_dir(output_dir / "figures")
    tables_dir = ensure_dir(output_dir / "tables")

    logger.info("Starting multi-strategy comparison experiment …")

    comparison = StrategyComparison(cfg=cfg)
    result = comparison.run(
        strategies=args.strategies,
        n_episodes_per_strategy=args.n_episodes,
        output_dir=output_dir,
    )

    # Build summary DataFrame
    import pandas as pd
    df = pd.DataFrame([s.model_dump() for s in result.summaries])

    if df.empty:
        logger.warning("No results generated.")
        return

    # Print leaderboard
    print("\n=== STRATEGY LEADERBOARD ===")
    leaderboard = (
        df.groupby("agent_strategy")[["composite_score", "ethical_alignment_score", "utility_score"]]
        .mean()
        .round(3)
        .sort_values("composite_score", ascending=False)
    )
    print(leaderboard.to_string())
    print(f"\nTotal episodes: {len(df)}")

    if not args.no_plots:
        logger.info("Generating figures …")
        try:
            plot_strategy_comparison(
                df, output_path=figures_dir / "strategy_comparison.png"
            )
            logger.info("Saved strategy_comparison.png")

            plot_metric_distributions(
                df,
                metric="composite_score",
                output_path=figures_dir / "composite_score_distribution.png",
            )
            logger.info("Saved composite_score_distribution.png")

            plot_score_heatmap(
                df, output_path=figures_dir / "score_heatmap.png"
            )
            logger.info("Saved score_heatmap.png")

            fig_radar = plot_radar_chart(df)
            fig_radar.write_html(str(figures_dir / "radar_chart.html"))
            logger.info("Saved radar_chart.html")

        except Exception as exc:  # noqa: BLE001
            logger.warning("Figure generation failed: %s", exc)

    logger.info("Experiment complete. Results in %s", output_dir)


if __name__ == "__main__":
    main()
