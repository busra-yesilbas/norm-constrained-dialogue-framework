"""
Run evaluation on saved episode transcripts.

Loads saved episode JSON files from results/sample_runs/ and produces
aggregated evaluation summaries as CSV and JSON.

Usage
-----
    python scripts/run_evaluation.py [--input-dir results/sample_runs]
                                     [--output-dir results/tables]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from norm_dialogue_framework.config import load_config
from norm_dialogue_framework.evaluation.evaluator import Evaluator
from norm_dialogue_framework.schemas import DialogueEpisode
from norm_dialogue_framework.utils import get_logger, set_seed

logger = get_logger("run_evaluation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved dialogue episode transcripts.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input-dir", default="results/sample_runs")
    parser.add_argument("--output-dir", default="results/tables")
    parser.add_argument("--prefix", default="evaluation_summary")
    return parser.parse_args()


def load_episodes(input_dir: Path) -> list[DialogueEpisode]:
    """Load all episode JSON files from a directory."""
    episodes = []
    files = list(input_dir.glob("episode_*.json"))
    if not files:
        logger.warning("No episode files found in %s", input_dir)
        return episodes

    for f in files:
        try:
            with f.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            episode = DialogueEpisode(**data)
            episodes.append(episode)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load %s: %s", f, exc)

    logger.info("Loaded %d episodes from %s", len(episodes), input_dir)
    return episodes


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.project.random_seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    episodes = load_episodes(input_dir)
    if not episodes:
        logger.error("No episodes to evaluate. Run run_simulation.py or run_experiments.py first.")
        sys.exit(1)

    evaluator = Evaluator()
    logger.info("Evaluating %d episodes …", len(episodes))
    summaries = evaluator.evaluate_batch(episodes)

    evaluator.save_summaries(summaries, output_dir=output_dir, prefix=args.prefix)

    # Print summary table
    df = evaluator.summaries_to_dataframe(summaries)
    if not df.empty and "agent_strategy" in df.columns:
        print("\n--- Strategy-Level Summary ---")
        agg = df.groupby("agent_strategy")[
            ["composite_score", "ethical_alignment_score", "utility_score"]
        ].mean().round(3)
        print(agg.to_string())
        print(f"\nTotal episodes evaluated: {len(summaries)}")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
