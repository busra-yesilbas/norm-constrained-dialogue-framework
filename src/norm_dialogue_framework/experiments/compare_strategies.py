"""
Strategy comparison pipeline.

Runs multiple dialogue episodes across all configured agent strategies
and respondent profiles, then collects and compares evaluation metrics.

The pipeline is designed for reproducibility:
  - Fixed random seed per episode
  - Deterministic scenario assignment
  - All outputs saved to results/
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from norm_dialogue_framework.agents.base_agent import BaseAgent
from norm_dialogue_framework.agents.baseline_agent import BaselineAgent
from norm_dialogue_framework.agents.candidate_selector_agent import CandidateSelectorAgent
from norm_dialogue_framework.agents.constrained_filter import ConstrainedFilterAgent
from norm_dialogue_framework.agents.critique_revise_agent import CritiqueReviseAgent
from norm_dialogue_framework.agents.rule_augmented_agent import RuleAugmentedAgent
from norm_dialogue_framework.config import FrameworkConfig
from norm_dialogue_framework.data.synthetic_generator import SyntheticScenarioGenerator
from norm_dialogue_framework.evaluation.evaluator import Evaluator
from norm_dialogue_framework.schemas import EpisodeSummary, ExperimentResult, Scenario
from norm_dialogue_framework.simulation.dialogue_runner import DialogueRunner
from norm_dialogue_framework.utils import ensure_dir, get_logger, save_json, set_seed

logger = get_logger(__name__)


def _build_agent(strategy: str, cfg: FrameworkConfig, seed: int) -> BaseAgent:
    """Instantiate an agent by strategy name."""
    llm_backend = cfg.agents.llm_backend
    if strategy == "baseline":
        return BaselineAgent(seed=seed, llm_backend=llm_backend)
    if strategy == "rule_augmented":
        return RuleAugmentedAgent(seed=seed, llm_backend=llm_backend)
    if strategy == "critique_revise":
        return CritiqueReviseAgent(
            max_revisions=cfg.agents.critique_revise.max_revisions,
            seed=seed,
            llm_backend=llm_backend,
        )
    if strategy == "candidate_selector":
        return CandidateSelectorAgent(
            n_candidates=cfg.agents.candidate_selector.n_candidates,
            seed=seed,
            llm_backend=llm_backend,
        )
    if strategy == "constrained_filter":
        return ConstrainedFilterAgent(seed=seed, llm_backend=llm_backend)
    raise ValueError(f"Unknown strategy: {strategy!r}")


class StrategyComparison:
    """Runs and compares multiple alignment strategies across episodes.

    Parameters
    ----------
    cfg:
        Framework configuration.
    scenarios:
        Optional pre-generated list of scenarios.  If None, generates
        fresh scenarios using the data config.
    """

    def __init__(
        self,
        cfg: FrameworkConfig,
        scenarios: Optional[list[Scenario]] = None,
    ) -> None:
        self._cfg = cfg
        self._scenarios = scenarios
        self._evaluator = Evaluator()

    def run(
        self,
        strategies: Optional[list[str]] = None,
        n_episodes_per_strategy: Optional[int] = None,
        output_dir: str | Path = "results",
    ) -> ExperimentResult:
        """Run the comparison experiment.

        Parameters
        ----------
        strategies:
            List of strategy names to compare.  Defaults to all configured strategies.
        n_episodes_per_strategy:
            Number of episodes per strategy.  Defaults to config value.
        output_dir:
            Root directory for saving results.

        Returns
        -------
        ExperimentResult
            Container with all episode summaries.
        """
        set_seed(self._cfg.project.random_seed)

        strategies = strategies or self._cfg.agents.strategies
        n_eps = n_episodes_per_strategy or self._cfg.experiments.n_episodes_per_strategy

        logger.info(
            "Starting strategy comparison: %d strategies × %d episodes = %d total",
            len(strategies),
            n_eps,
            len(strategies) * n_eps,
        )

        # Prepare scenarios
        scenarios = self._scenarios or self._generate_scenarios(n_eps)

        result = ExperimentResult()
        all_summaries: list[EpisodeSummary] = []

        for strategy in strategies:
            logger.info("--- Strategy: %s ---", strategy)
            strategy_summaries = self._run_strategy(strategy, scenarios[:n_eps], n_eps)
            all_summaries.extend(strategy_summaries)

        result.summaries = all_summaries
        result.n_episodes = len(all_summaries)

        # Save outputs
        out = ensure_dir(output_dir)
        self._save_results(result, out)

        logger.info(
            "Experiment complete: %d episodes evaluated.", result.n_episodes
        )
        return result

    def summarise(self, result: ExperimentResult) -> pd.DataFrame:
        """Return a strategy-level summary DataFrame from an experiment result."""
        df = pd.DataFrame([s.model_dump() for s in result.summaries])
        if df.empty:
            return df

        agg_cols = [
            "ethical_alignment_score",
            "utility_score",
            "composite_score",
            "mean_coercion_risk",
            "mean_leading_question_risk",
            "mean_empathy_score",
            "mean_information_yield",
            "final_trust_level",
            "final_stress_level",
        ]
        available = [c for c in agg_cols if c in df.columns]
        summary = (
            df.groupby("agent_strategy")[available]
            .agg(["mean", "std"])
            .round(4)
        )
        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_scenarios(self, n: int) -> list[Scenario]:
        gen = SyntheticScenarioGenerator(
            seed=self._cfg.project.random_seed,
            scenario_types=self._cfg.data.scenario_types or None,
            respondent_profiles=self._cfg.data.respondent_profiles or None,
        )
        return gen.generate(n=max(n, 10))

    def _run_strategy(
        self,
        strategy: str,
        scenarios: list[Scenario],
        n_eps: int,
    ) -> list[EpisodeSummary]:
        summaries: list[EpisodeSummary] = []
        episode_seed_base = self._cfg.project.random_seed

        for i, scenario in enumerate(
            tqdm(scenarios[:n_eps], desc=f"  {strategy}", leave=False)
        ):
            ep_seed = episode_seed_base + i * 100
            agent = _build_agent(strategy, self._cfg, seed=ep_seed)
            runner = DialogueRunner(
                agent=agent,
                max_turns=self._cfg.simulation.max_turns,
                min_turns=self._cfg.simulation.min_turns,
                seed=ep_seed,
                evaluator=self._evaluator,
            )
            t0 = time.perf_counter()
            episode = runner.run(scenario)
            elapsed = time.perf_counter() - t0

            summary = self._evaluator.evaluate_episode(episode)
            summaries.append(summary)

            logger.debug(
                "  ep %02d/%02d [%s] composite=%.3f (%.2fs)",
                i + 1,
                n_eps,
                strategy,
                summary.composite_score,
                elapsed,
            )

        return summaries

    def _save_results(self, result: ExperimentResult, out: Path) -> None:
        tables_dir = ensure_dir(out / "tables")
        df = pd.DataFrame([s.model_dump() for s in result.summaries])
        if not df.empty:
            df.to_csv(tables_dir / "experiment_summary.csv", index=False)
            logger.info("Saved experiment_summary.csv → %s", tables_dir)

        save_json(
            result.model_dump(mode="json"),
            out / "sample_runs" / f"experiment_{result.experiment_id}.json",
        )
