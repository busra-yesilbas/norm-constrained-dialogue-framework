"""
Generate a synthetic scenario dataset.

Usage
-----
    python scripts/generate_dataset.py [--config configs/default.yaml] [--n 100]

Outputs
-------
    data/synthetic/scenarios.json
    data/synthetic/scenarios.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from norm_dialogue_framework.config import load_config
from norm_dialogue_framework.data.synthetic_generator import SyntheticScenarioGenerator
from norm_dialogue_framework.utils import get_logger, set_seed

logger = get_logger("generate_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic dialogue scenarios.")
    parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to config YAML."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of scenarios to generate (overrides config).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (overrides config).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed = args.seed if args.seed is not None else cfg.project.random_seed
    set_seed(seed)

    n = args.n if args.n is not None else cfg.data.n_scenarios
    output_dir = args.output_dir or cfg.data.output_dir

    logger.info("Generating %d synthetic scenarios (seed=%d) …", n, seed)

    generator = SyntheticScenarioGenerator(
        seed=seed,
        scenario_types=cfg.data.scenario_types or None,
        respondent_profiles=cfg.data.respondent_profiles or None,
    )
    scenarios = generator.generate(n=n)
    generator.save(scenarios, output_dir=output_dir)

    logger.info("Done. Scenarios saved to %s", output_dir)

    # Print a quick summary
    df = generator.to_dataframe(scenarios)
    print("\n--- Dataset Summary ---")
    print(df.groupby(["scenario_type", "respondent_profile"]).size().to_string())
    print(f"\nTotal: {len(scenarios)} scenarios")


if __name__ == "__main__":
    main()
