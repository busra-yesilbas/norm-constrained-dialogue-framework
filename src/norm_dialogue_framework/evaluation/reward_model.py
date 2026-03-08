"""
Lightweight reward model for alignment optimisation experiments.

Implements a simple offline reward function that can be used to score
candidate utterances during candidate-selection or re-ranking.

IMPORTANT DISCLAIMER:
    This is an experimental research approximation of alignment reward
    modelling.  It is NOT:
    - A trained neural reward model
    - A production RL/RLHF system
    - A clinical or legal assessment tool

    It uses heuristic rules and configurable weights to simulate
    reward-based selection at a conceptual level only.
    See: methodology.md for full limitations.
"""

from __future__ import annotations

from typing import Optional

from norm_dialogue_framework.agents.base_agent import DialogueState
from norm_dialogue_framework.evaluation.rule_checks import RuleChecker
from norm_dialogue_framework.evaluation.metrics import MetricsComputer
from norm_dialogue_framework.schemas import TurnMetrics
from norm_dialogue_framework.utils import clamp, get_logger

logger = get_logger(__name__)


class RewardModel:
    """Heuristic reward model for scoring candidate utterances.

    Combines ethical alignment and utility scores into a composite
    reward signal.  Can be used to rank candidate responses during
    Best-of-N sampling.

    Parameters
    ----------
    ethical_weight:
        Weight assigned to the ethical alignment component.
    utility_weight:
        Weight assigned to the utility component.
    norm_violation_penalty:
        Additional penalty subtracted per detected norm violation.
    """

    def __init__(
        self,
        ethical_weight: float = 0.55,
        utility_weight: float = 0.45,
        norm_violation_penalty: float = 0.15,
    ) -> None:
        self._eth_w = ethical_weight
        self._util_w = utility_weight
        self._penalty = norm_violation_penalty
        self._metrics = MetricsComputer(
            ethical_composite_weight=ethical_weight,
            utility_composite_weight=utility_weight,
        )

    def score(self, utterance: str, state: DialogueState) -> float:
        """Score a single candidate utterance.

        Parameters
        ----------
        utterance:
            Candidate utterance text.
        state:
            Current dialogue state.

        Returns
        -------
        float
            Composite reward in [0, 1].  Higher is better.
        """
        checker = RuleChecker()
        rule_result = checker.check(utterance)
        metrics = self._metrics.compute_turn_metrics(
            utterance, rule_result, state, previous_trust=state.trust_level
        )
        reward = metrics.composite_score

        # Apply hard penalties for critical violations
        n_violations = 0
        if rule_result.coercion_risk > 0.7:
            n_violations += 1
        if rule_result.leading_question_risk > 0.7:
            n_violations += 1
        if rule_result.pressure_escalation_risk > 0.7:
            n_violations += 1

        reward = clamp(reward - n_violations * self._penalty)
        return reward

    def rank_candidates(
        self, candidates: list[str], state: DialogueState
    ) -> list[tuple[str, float]]:
        """Rank a list of candidate utterances by reward score.

        Parameters
        ----------
        candidates:
            List of candidate utterance strings.
        state:
            Current dialogue state.

        Returns
        -------
        list[tuple[str, float]]
            Sorted list of (utterance, score) pairs, highest first.
        """
        scored = [(c, self.score(c, state)) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def get_detailed_scores(
        self, utterance: str, state: DialogueState
    ) -> dict[str, float]:
        """Return a detailed breakdown of reward components.

        Parameters
        ----------
        utterance:
            Candidate utterance.
        state:
            Current dialogue state.

        Returns
        -------
        dict[str, float]
            Mapping of metric names to scores.
        """
        checker = RuleChecker()
        rule_result = checker.check(utterance)
        metrics = self._metrics.compute_turn_metrics(
            utterance, rule_result, state, previous_trust=state.trust_level
        )
        return {
            "composite_reward": self.score(utterance, state),
            "ethical_alignment_score": metrics.ethical_alignment_score,
            "utility_score": metrics.utility_score,
            "coercion_risk": metrics.coercion_risk,
            "leading_question_risk": metrics.leading_question_risk,
            "empathy_score": metrics.empathy_score,
            "neutrality_score": metrics.neutrality_score,
            "information_yield": metrics.information_yield,
            "trust_preservation": metrics.trust_preservation,
        }
