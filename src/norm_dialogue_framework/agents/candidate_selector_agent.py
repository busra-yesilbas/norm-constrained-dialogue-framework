"""
Candidate-Generation + Reward-Based Selector Agent.

Generates N diverse candidate utterances and scores each using a
composite reward function that balances utility and norm compliance.
The highest-scoring candidate is selected as the agent's turn.

This implements a lightweight offline reward optimisation workflow,
analogous to Best-of-N sampling or rejection sampling used in RLHF
pipelines.  It is an experimental approximation only and is not a
production RL system.

Reference framing:
    Stiennon et al. (2020) – Learning to summarize with human feedback
    Ouyang et al. (2022) – Training language models to follow instructions with human feedback
"""

from __future__ import annotations

from norm_dialogue_framework.agents.base_agent import (
    BaseAgent,
    DialogueState,
    _FOLLOWUP_QUESTIONS,
    _OPENING_QUESTIONS,
    _CLARIFICATION_QUESTIONS,
    _CLOSING_QUESTIONS,
    _extract_topic,
)
from norm_dialogue_framework.agents.rule_augmented_agent import _norm_score
from norm_dialogue_framework.utils import get_logger, clamp

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Utility heuristics
# ---------------------------------------------------------------------------

_OPEN_QUESTION_SIGNALS = ["what", "how", "could you", "can you", "describe", "tell me"]
_CLOSED_QUESTION_SIGNALS = ["did you", "were you", "is it", "was it", "have you"]


def _utility_score(utterance: str, state: DialogueState) -> float:
    """Estimate information-gathering utility of an utterance (0–1)."""
    u = utterance.lower()
    score = 0.5

    # Open-ended questions yield more information
    if any(sig in u for sig in _OPEN_QUESTION_SIGNALS):
        score += 0.20
    if any(sig in u for sig in _CLOSED_QUESTION_SIGNALS):
        score -= 0.10

    # Addressing unresolved goals is valuable
    unresolved = [g for g in state.scenario.communication_goals if g not in state.goals_addressed]
    for goal in unresolved:
        keywords = goal.lower().split()
        if any(kw in u for kw in keywords if len(kw) > 3):
            score += 0.15
            break

    # Shorter, focused questions are preferable
    word_count = len(utterance.split())
    if word_count > 40:
        score -= 0.10
    elif word_count < 8:
        score -= 0.05

    return clamp(score)


def _composite_reward(
    utterance: str,
    state: DialogueState,
    ethical_weight: float = 0.55,
    utility_weight: float = 0.45,
) -> float:
    """Compute a composite reward for an utterance.

    Parameters
    ----------
    utterance:
        Candidate utterance to score.
    state:
        Current dialogue state.
    ethical_weight:
        Weight assigned to the ethical alignment score.
    utility_weight:
        Weight assigned to the utility score.
    """
    norm_s = _norm_score(utterance, state.scenario.recommended_norms)
    util_s = _utility_score(utterance, state)
    return clamp(ethical_weight * norm_s + utility_weight * util_s)


class CandidateSelectorAgent(BaseAgent):
    """Agent that generates N candidates and selects the best by reward.

    This agent explicitly models a Best-of-N sampling alignment strategy.
    It generates a diverse candidate pool from templates (or an LLM),
    scores each with a composite ethical + utility reward, and returns
    the highest-scoring candidate.

    This is documented as an experimental research approximation and
    does not constitute a production RL training system.

    Parameters
    ----------
    n_candidates:
        Number of candidates to generate per turn.
    ethical_weight, utility_weight:
        Reward function mixing weights.
    seed, llm_backend:
        Passed through to BaseAgent.
    """

    def __init__(
        self,
        n_candidates: int = 5,
        ethical_weight: float = 0.55,
        utility_weight: float = 0.45,
        seed: int = 42,
        llm_backend=None,
    ) -> None:
        super().__init__(seed=seed, llm_backend=llm_backend)
        self._strategy_name = "candidate_selector"
        self._n_candidates = n_candidates
        self._ethical_weight = ethical_weight
        self._utility_weight = utility_weight

    def generate_next_turn(self, state: DialogueState) -> str:
        candidates = self._generate_candidate_pool(state)
        scored = [
            (c, _composite_reward(c, state, self._ethical_weight, self._utility_weight))
            for c in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        best, best_score = scored[0]
        logger.debug(
            "CandidateSelector: selected from %d candidates (reward=%.3f).",
            len(candidates),
            best_score,
        )
        return best

    def _generate_candidate_pool(self, state: DialogueState) -> list[str]:
        n = state.n_agent_turns
        last_resp = state.last_respondent_utterance or ""
        topic = _extract_topic(last_resp)

        if n == 0:
            pool = _OPENING_QUESTIONS
        else:
            followups = [q.replace("{topic}", topic) for q in _FOLLOWUP_QUESTIONS]
            clarifications = [
                q.replace("{paraphrase}", last_resp[:60]).replace("{event}", topic)
                for q in _CLARIFICATION_QUESTIONS
            ]
            pool = followups + clarifications + _CLOSING_QUESTIONS

        k = min(self._n_candidates, len(pool))
        return self._rng.sample(pool, k=k)

    def get_all_candidate_scores(
        self, state: DialogueState
    ) -> list[dict]:
        """Return all candidates with their reward scores (for analysis)."""
        candidates = self._generate_candidate_pool(state)
        results = []
        for c in candidates:
            reward = _composite_reward(c, state, self._ethical_weight, self._utility_weight)
            norm_s = _norm_score(c, state.scenario.recommended_norms)
            util_s = _utility_score(c, state)
            results.append(
                {
                    "utterance": c,
                    "norm_score": round(norm_s, 3),
                    "utility_score": round(util_s, 3),
                    "composite_reward": round(reward, 3),
                }
            )
        results.sort(key=lambda x: x["composite_reward"], reverse=True)
        return results
