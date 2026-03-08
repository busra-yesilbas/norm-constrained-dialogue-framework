"""
Constrained Response Filter Agent.

Applies a hard constraint filter: any generated utterance that violates
one or more critical norms is rejected and replaced with a safe fallback
from a curated norm-compliant response library.

This strategy represents the most conservative alignment approach:
zero-tolerance for norm violations, at the cost of potentially lower
information yield when the primary response is frequently rejected.

The filter can be composed with any other agent strategy as a wrapper.
"""

from __future__ import annotations

from norm_dialogue_framework.agents.base_agent import BaseAgent, DialogueState
from norm_dialogue_framework.agents.critique_revise_agent import _critique
from norm_dialogue_framework.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Curated norm-compliant fallback library
# ---------------------------------------------------------------------------

_SAFE_FALLBACKS: list[str] = [
    "Thank you for sharing that. Could you tell me more about what happened, in your own words?",
    "I appreciate you taking the time to explain. Is there anything else you'd like to add?",
    "That's helpful context. Could you walk me through the sequence of events as you remember them?",
    "I want to make sure I understand correctly. In your own words, what was the situation like for you?",
    "Take your time. What else can you tell me about that period?",
    "I'd like to understand your perspective fully. Could you describe what you observed?",
    "Thank you. Could you help me understand what was going through your mind at that point?",
    "Is there anything about the situation that you think is important for me to know?",
    "What would you say was the most significant part of what happened?",
    "How did things unfold from your perspective?",
]

_CRITICAL_VIOLATION_THRESHOLD = 1  # Reject if at least this many violations found


class ConstrainedFilterAgent(BaseAgent):
    """Agent that applies a hard constraint filter on generated responses.

    Wraps a base generation strategy (defaults to template selection) with
    a strict post-generation filter.  Any utterance containing one or more
    norm violations is discarded and replaced with a pre-approved safe fallback.

    This is the most conservative alignment strategy, prioritising ethical
    safety over information yield.

    Parameters
    ----------
    seed, llm_backend:
        Passed through to BaseAgent.
    """

    def __init__(self, seed: int = 42, llm_backend=None) -> None:
        super().__init__(seed=seed, llm_backend=llm_backend)
        self._strategy_name = "constrained_filter"
        self._fallback_index = 0

    def generate_next_turn(self, state: DialogueState) -> str:
        candidate = self._pick_question(state)
        norms = state.scenario.recommended_norms

        if self._llm_backend:
            llm_candidate = self._call_llm(self._build_prompt(state))
            if llm_candidate:
                candidate = llm_candidate

        issues = _critique(candidate, norms)
        if len(issues) >= _CRITICAL_VIOLATION_THRESHOLD:
            logger.debug(
                "ConstrainedFilter: rejected candidate (violations: %s). Using safe fallback.",
                issues,
            )
            return self._next_safe_fallback()

        logger.debug("ConstrainedFilter: candidate passed constraint filter.")
        return candidate

    def _next_safe_fallback(self) -> str:
        """Cycle through the safe fallback library."""
        fallback = _SAFE_FALLBACKS[self._fallback_index % len(_SAFE_FALLBACKS)]
        self._fallback_index += 1
        return fallback

    def _build_prompt(self, state: DialogueState) -> str:
        norms_str = ", ".join(state.scenario.recommended_norms)
        context = state.scenario.context_summary
        history_lines = "\n".join(
            f"{t.speaker.upper()}: {t.utterance}" for t in state.history[-4:]
        )
        return (
            f"You are a strictly norm-compliant conversational agent. "
            f"You must adhere to: {norms_str}.\n\n"
            f"Context: {context}\n\n"
            f"Recent dialogue:\n{history_lines}\n\n"
            f"Generate a single, short, open-ended, non-coercive, non-leading question. "
            f"Return only the question text."
        )
