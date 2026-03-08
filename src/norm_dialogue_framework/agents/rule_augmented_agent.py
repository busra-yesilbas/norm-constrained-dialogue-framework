"""
Rule-Augmented Dialogue Agent.

Generates candidate utterances using the baseline template approach and
then applies a lightweight rule-based filter to select the utterance
that best adheres to the scenario's recommended norms.

When an LLM backend is configured, norm-aware instructions are injected
into the system prompt before generation.
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
from norm_dialogue_framework.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Norm violation heuristics
# ---------------------------------------------------------------------------

_COERCIVE_PATTERNS: list[str] = [
    "you must",
    "you have to",
    "you need to tell me",
    "don't lie",
    "be honest",
    "i know you",
    "we already know",
    "admit",
    "confess",
]

_LEADING_PATTERNS: list[str] = [
    "wasn't it",
    "isn't it",
    "don't you think",
    "surely you",
    "obviously",
    "clearly you",
    "you said earlier that",
    "as you mentioned",
]


def _norm_score(utterance: str, norms: list[str]) -> float:
    """Heuristic norm compliance score for an utterance (0–1)."""
    u = utterance.lower()
    score = 1.0

    if "non_coercion" in norms:
        for p in _COERCIVE_PATTERNS:
            if p in u:
                score -= 0.25

    if "non_leading" in norms:
        for p in _LEADING_PATTERNS:
            if p in u:
                score -= 0.20

    if "empathy" in norms:
        empathy_phrases = ["i understand", "that must be", "thank you for sharing", "i appreciate"]
        if any(ep in u for ep in empathy_phrases):
            score += 0.10

    if "transparency" in norms:
        transparency_phrases = ["the purpose of this", "i'm asking because", "to help me understand"]
        if any(tp in u for tp in transparency_phrases):
            score += 0.05

    return max(0.0, min(1.0, score))


class RuleAugmentedAgent(BaseAgent):
    """Agent that applies explicit norm rules during generation.

    Generates a small pool of candidate utterances from templates and
    scores each against the scenario's recommended norms.  The highest-
    scoring candidate is selected.

    Parameters
    ----------
    n_candidates:
        Number of template candidates to generate and score.
    seed, llm_backend:
        Passed through to BaseAgent.
    """

    def __init__(
        self, n_candidates: int = 4, seed: int = 42, llm_backend=None
    ) -> None:
        super().__init__(seed=seed, llm_backend=llm_backend)
        self._strategy_name = "rule_augmented"
        self._n_candidates = n_candidates

    def generate_next_turn(self, state: DialogueState) -> str:
        if self._llm_backend:
            prompt = self._build_prompt(state)
            result = self._call_llm(prompt)
            if result:
                return result

        candidates = self._generate_candidates(state)
        norms = state.scenario.recommended_norms
        scored = [(c, _norm_score(c, norms)) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        best = scored[0][0]
        logger.debug(
            "RuleAugmented: selected utterance with norm score %.2f", scored[0][1]
        )
        return best

    def _generate_candidates(self, state: DialogueState) -> list[str]:
        """Generate a diverse pool of candidate utterances."""
        candidates: list[str] = []
        n = state.n_agent_turns
        last_resp = state.last_respondent_utterance or ""

        if n == 0:
            candidates.extend(self._rng.sample(_OPENING_QUESTIONS, k=min(self._n_candidates, len(_OPENING_QUESTIONS))))
        else:
            topic = _extract_topic(last_resp)
            followups = [q.replace("{topic}", topic) for q in _FOLLOWUP_QUESTIONS]
            clarifications = [
                q.replace("{paraphrase}", last_resp[:60]).replace("{event}", topic)
                for q in _CLARIFICATION_QUESTIONS
            ]
            closing = _CLOSING_QUESTIONS

            pool = followups + clarifications + closing
            candidates.extend(self._rng.sample(pool, k=min(self._n_candidates, len(pool))))

        if len(candidates) < 2:
            candidates.append(self._pick_question(state))

        return candidates

    def _build_prompt(self, state: DialogueState) -> str:
        norms_str = ", ".join(state.scenario.recommended_norms)
        context = state.scenario.context_summary
        history_lines = "\n".join(
            f"{t.speaker.upper()}: {t.utterance}" for t in state.history[-6:]
        )
        return (
            f"You are a norm-constrained conversational agent. You must strictly adhere to "
            f"the following norms: {norms_str}.\n\n"
            f"Context: {context}\n\n"
            f"Recent dialogue:\n{history_lines}\n\n"
            f"Generate the agent's next utterance. Avoid leading questions, coercive language, "
            f"and pressure. Use open-ended, neutral language."
        )
