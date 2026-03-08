"""
Run a single dialogue simulation episode.

Usage
-----
    python scripts/run_simulation.py [--strategy baseline] [--scenario-type witness_recall]
                                     [--profile anxious] [--turns 8] [--verbose]

Outputs
-------
    results/sample_runs/episode_<id>.json
    Transcript printed to stdout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from norm_dialogue_framework.agents import (
    BaselineAgent,
    CandidateSelectorAgent,
    ConstrainedFilterAgent,
    CritiqueReviseAgent,
    RuleAugmentedAgent,
)
from norm_dialogue_framework.config import load_config
from norm_dialogue_framework.data.synthetic_generator import SyntheticScenarioGenerator
from norm_dialogue_framework.evaluation.evaluator import Evaluator
from norm_dialogue_framework.schemas import RespondentProfile, ScenarioType
from norm_dialogue_framework.simulation.dialogue_runner import DialogueRunner
from norm_dialogue_framework.utils import get_logger, set_seed

logger = get_logger("run_simulation")

_AGENT_MAP = {
    "baseline": BaselineAgent,
    "rule_augmented": RuleAugmentedAgent,
    "critique_revise": CritiqueReviseAgent,
    "candidate_selector": CandidateSelectorAgent,
    "constrained_filter": ConstrainedFilterAgent,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single dialogue simulation.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--strategy",
        choices=list(_AGENT_MAP.keys()),
        default="candidate_selector",
    )
    parser.add_argument("--scenario-type", default=None, choices=[e.value for e in ScenarioType])
    parser.add_argument("--profile", default=None, choices=[e.value for e in RespondentProfile])
    parser.add_argument("--turns", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-dir", default="results/sample_runs")
    return parser.parse_args()


def print_transcript(episode) -> None:
    print("\n" + "=" * 70)
    print(f"EPISODE: {episode.episode_id}")
    print(f"Strategy: {episode.agent_strategy}")
    print(f"Scenario: {episode.scenario.scenario_type} | Profile: {episode.scenario.respondent_profile}")
    print(f"Sensitivity: {episode.scenario.sensitivity_level}")
    print(f"\nContext:\n  {episode.scenario.context_summary}")
    print("=" * 70 + "\n")

    for turn in episode.turns:
        speaker = turn.speaker.upper() if hasattr(turn.speaker, "upper") else str(turn.speaker).upper()
        print(f"[{speaker}] {turn.utterance}")
        if turn.turn_metrics:
            m = turn.turn_metrics
            print(
                f"  -> composite={m.composite_score:.2f} | "
                f"ethical={m.ethical_alignment_score:.2f} | "
                f"utility={m.utility_score:.2f} | "
                f"coercion_risk={m.coercion_risk:.2f}"
            )
        print()

    print("=" * 70)
    print(f"Final trust: {episode.final_trust_level:.2f} | Final stress: {episode.final_stress_level:.2f}")
    print(f"Total agent turns: {episode.total_turns}")
    print("=" * 70 + "\n")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.project.random_seed
    set_seed(seed)

    # Generate a scenario
    scenario_types = [args.scenario_type] if args.scenario_type else None
    profiles = [args.profile] if args.profile else None
    gen = SyntheticScenarioGenerator(
        seed=seed,
        scenario_types=scenario_types,
        respondent_profiles=profiles,
    )
    scenario = gen.generate(n=1)[0]
    logger.info("Scenario: %s | Profile: %s", scenario.scenario_type, scenario.respondent_profile)

    # Build agent
    agent_cls = _AGENT_MAP[args.strategy]
    agent = agent_cls(seed=seed)

    # Build evaluator
    evaluator = Evaluator()

    # Build runner
    max_turns = args.turns or cfg.simulation.max_turns
    runner = DialogueRunner(
        agent=agent,
        max_turns=max_turns,
        min_turns=cfg.simulation.min_turns,
        seed=seed,
        evaluator=evaluator,
    )

    # Run
    logger.info("Running simulation [strategy=%s] …", args.strategy)
    episode = runner.run(scenario)

    # Evaluate
    summary = evaluator.evaluate_episode(episode)

    # Display transcript
    print_transcript(episode)

    # Print summary
    print("--- EPISODE SUMMARY ---")
    print(f"Composite score:          {summary.composite_score:.3f}")
    print(f"Ethical alignment score:  {summary.ethical_alignment_score:.3f}")
    print(f"Utility score:            {summary.utility_score:.3f}")
    print(f"Mean coercion risk:       {summary.mean_coercion_risk:.3f}")
    print(f"Mean empathy score:       {summary.mean_empathy_score:.3f}")
    print(f"Mean information yield:   {summary.mean_information_yield:.3f}")

    # Save
    evaluator.save_transcript(episode, output_dir=args.output_dir)
    logger.info("Transcript saved.")


if __name__ == "__main__":
    main()
