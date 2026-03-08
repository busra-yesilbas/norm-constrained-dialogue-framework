"""
End-to-end evaluator for dialogue episodes.

Combines rule checks and metrics computation to produce per-turn
evaluations and episode-level summaries.  Supports both online
(turn-by-turn) and offline (post-hoc on saved transcripts) evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from norm_dialogue_framework.agents.base_agent import DialogueState
from norm_dialogue_framework.evaluation.metrics import MetricsComputer
from norm_dialogue_framework.evaluation.rule_checks import RuleChecker
from norm_dialogue_framework.schemas import (
    DialogueEpisode,
    DialogueTurn,
    EpisodeSummary,
    TurnMetrics,
    Speaker,
)
from norm_dialogue_framework.utils import clamp, ensure_dir, get_logger, save_json

logger = get_logger(__name__)


class Evaluator:
    """Evaluates dialogue episodes for norm compliance and utility.

    Can be used:
    1. Online: passed to DialogueRunner to evaluate each turn in real time.
    2. Offline: called with a completed DialogueEpisode object.

    Parameters
    ----------
    metrics_weights:
        Optional override for metric weights.
    """

    def __init__(self, metrics_weights: Optional[dict] = None) -> None:
        self._checker = RuleChecker()
        self._metrics = MetricsComputer()
        self._previous_trust: Optional[float] = None

    def evaluate_turn(
        self, turn: DialogueTurn, state: DialogueState
    ) -> TurnMetrics:
        """Evaluate a single agent turn in real time.

        Parameters
        ----------
        turn:
            The agent's turn to evaluate.
        state:
            Current dialogue state.

        Returns
        -------
        TurnMetrics
            Computed metrics for this turn.
        """
        rule_result = self._checker.check(turn.utterance)
        metrics = self._metrics.compute_turn_metrics(
            turn.utterance, rule_result, state, self._previous_trust
        )
        self._previous_trust = state.trust_level
        return metrics

    def evaluate_episode(self, episode: DialogueEpisode) -> EpisodeSummary:
        """Evaluate a completed dialogue episode.

        Runs all checks on each agent turn and computes aggregated
        episode-level statistics.

        Parameters
        ----------
        episode:
            Completed dialogue episode.

        Returns
        -------
        EpisodeSummary
            Aggregated episode-level metrics.
        """
        self._checker.reset_history()
        self._previous_trust = None

        agent_turns = [t for t in episode.turns if t.speaker == Speaker.AGENT]
        if not agent_turns:
            logger.warning("Episode %s has no agent turns.", episode.episode_id)
            return self._empty_summary(episode)

        metrics_list: list[TurnMetrics] = []
        prev_trust: Optional[float] = None

        for i, turn in enumerate(agent_turns):
            rule_result = self._checker.check(turn.utterance)

            # Build a minimal state for metrics computation
            trust = (
                turn.respondent_trust_level
                if turn.respondent_trust_level is not None
                else 0.5
            )
            stress = (
                turn.respondent_stress_level
                if turn.respondent_stress_level is not None
                else 0.3
            )

            from norm_dialogue_framework.agents.base_agent import DialogueState as DS
            temp_state = DS(
                scenario=episode.scenario,
                trust_level=trust,
                stress_level=stress,
                turn_number=i,
            )
            m = self._metrics.compute_turn_metrics(
                turn.utterance, rule_result, temp_state, prev_trust
            )
            turn.turn_metrics = m
            metrics_list.append(m)
            prev_trust = trust

        return self._aggregate(episode, metrics_list)

    def evaluate_batch(
        self, episodes: list[DialogueEpisode]
    ) -> list[EpisodeSummary]:
        """Evaluate a list of episodes and return summaries."""
        summaries = []
        for ep in episodes:
            try:
                s = self.evaluate_episode(ep)
                summaries.append(s)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to evaluate episode %s: %s", ep.episode_id, exc)
        return summaries

    def summaries_to_dataframe(self, summaries: list[EpisodeSummary]) -> pd.DataFrame:
        """Convert a list of EpisodeSummary objects to a pandas DataFrame."""
        return pd.DataFrame([s.model_dump() for s in summaries])

    def save_summaries(
        self,
        summaries: list[EpisodeSummary],
        output_dir: str | Path = "results/tables",
        prefix: str = "evaluation_summary",
    ) -> None:
        """Save summaries as CSV and JSON."""
        out = ensure_dir(output_dir)
        df = self.summaries_to_dataframe(summaries)
        df.to_csv(out / f"{prefix}.csv", index=False)
        save_json([s.model_dump(mode="json") for s in summaries], out / f"{prefix}.json")
        logger.info("Saved evaluation summaries → %s", out)

    def save_transcript(
        self,
        episode: DialogueEpisode,
        output_dir: str | Path = "results/sample_runs",
    ) -> None:
        """Save a full episode transcript as JSON."""
        out = ensure_dir(output_dir)
        path = out / f"episode_{episode.episode_id}.json"
        save_json(episode.model_dump(mode="json"), path)
        logger.info("Saved transcript → %s", path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _aggregate(
        self, episode: DialogueEpisode, metrics_list: list[TurnMetrics]
    ) -> EpisodeSummary:
        def mean(vals):
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        return EpisodeSummary(
            episode_id=episode.episode_id,
            case_id=episode.scenario.case_id,
            scenario_type=str(episode.scenario.scenario_type),
            respondent_profile=str(episode.scenario.respondent_profile),
            sensitivity_level=str(episode.scenario.sensitivity_level),
            agent_strategy=str(episode.agent_strategy),
            total_turns=episode.total_turns,
            mean_coercion_risk=mean([m.coercion_risk for m in metrics_list]),
            mean_leading_question_risk=mean([m.leading_question_risk for m in metrics_list]),
            mean_pressure_escalation_risk=mean([m.pressure_escalation_risk for m in metrics_list]),
            mean_empathy_score=mean([m.empathy_score for m in metrics_list]),
            mean_transparency_score=mean([m.transparency_score for m in metrics_list]),
            mean_neutrality_score=mean([m.neutrality_score for m in metrics_list]),
            mean_procedural_fairness_score=mean([m.procedural_fairness_score for m in metrics_list]),
            mean_information_yield=mean([m.information_yield for m in metrics_list]),
            mean_trust_preservation=mean([m.trust_preservation for m in metrics_list]),
            mean_engagement_score=mean([m.engagement_score for m in metrics_list]),
            ethical_alignment_score=mean([m.ethical_alignment_score for m in metrics_list]),
            utility_score=mean([m.utility_score for m in metrics_list]),
            composite_score=mean([m.composite_score for m in metrics_list]),
            final_trust_level=round(episode.final_trust_level or 0.0, 4),
            final_stress_level=round(episode.final_stress_level or 0.0, 4),
            completed=episode.completed,
        )

    def _empty_summary(self, episode: DialogueEpisode) -> EpisodeSummary:
        return EpisodeSummary(
            episode_id=episode.episode_id,
            case_id=episode.scenario.case_id,
            scenario_type=str(episode.scenario.scenario_type),
            respondent_profile=str(episode.scenario.respondent_profile),
            sensitivity_level=str(episode.scenario.sensitivity_level),
            agent_strategy=str(episode.agent_strategy),
            total_turns=0,
            mean_coercion_risk=0.0,
            mean_leading_question_risk=0.0,
            mean_pressure_escalation_risk=0.0,
            mean_empathy_score=0.0,
            mean_transparency_score=0.0,
            mean_neutrality_score=0.0,
            mean_procedural_fairness_score=0.0,
            mean_information_yield=0.0,
            mean_trust_preservation=0.0,
            mean_engagement_score=0.0,
            ethical_alignment_score=0.0,
            utility_score=0.0,
            composite_score=0.0,
            final_trust_level=0.0,
            final_stress_level=0.0,
            completed=False,
        )
