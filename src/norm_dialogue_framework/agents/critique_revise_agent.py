"""
Critique-and-Revise Dialogue Agent.

Generates an initial utterance, critiques it against norm requirements,
and iteratively revises it until it passes all checks or the maximum
number of revision cycles is reached.

In fallback mode, revision is implemented via a lightweight heuristic
rewrite system.  With an LLM backend, each critique and revision step
delegates to the language model.
"""

from __future__ import annotations

from norm_dialogue_framework.agents.base_agent import BaseAgent, DialogueState
from norm_dialogue_framework.agents.rule_augmented_agent import _norm_score, _COERCIVE_PATTERNS, _LEADING_PATTERNS
from norm_dialogue_framework.utils import get_logger, clamp

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Rewrite heuristics for fallback revision
# ---------------------------------------------------------------------------

_REWRITES: list[tuple[str, str]] = [
    ("you must", "would you be willing to"),
    ("you have to", "could you"),
    ("don't lie", "please share whatever you remember"),
    ("admit", "describe"),
    ("confess", "explain"),
    ("wasn't it", ""),
    ("isn't it", ""),
    ("don't you think", "what do you think"),
    ("surely you", "do you"),
    ("obviously", ""),
    ("clearly you", "it seems"),
    ("we already know", ""),
]


def _apply_rewrites(text: str) -> str:
    """Apply heuristic rewrites to reduce norm violations."""
    result = text
    for bad, good in _REWRITES:
        result = result.replace(bad, good)
    # Collapse double spaces
    while "  " in result:
        result = result.replace("  ", " ")
    return result.strip()


def _critique(utterance: str, norms: list[str]) -> list[str]:
    """Return a list of critique strings for any norm violations found."""
    u = utterance.lower()
    issues: list[str] = []

    if "non_coercion" in norms:
        for p in _COERCIVE_PATTERNS:
            if p in u:
                issues.append(f"Coercive language detected: '{p}'")

    if "non_leading" in norms:
        for p in _LEADING_PATTERNS:
            if p in u:
                issues.append(f"Leading language detected: '{p}'")

    return issues


class CritiqueReviseAgent(BaseAgent):
    """Agent that critiques and revises its own utterances.

    Implements a lightweight self-refinement loop:
    1. Generate initial candidate utterance.
    2. Critique for norm violations.
    3. Revise if violations found (up to ``max_revisions`` cycles).

    This mimics the Constitutional AI / RLHF critique-and-revision
    paradigm at a template level for demonstration purposes.

    Parameters
    ----------
    max_revisions:
        Maximum number of critique-revise cycles per turn.
    seed, llm_backend:
        Passed through to BaseAgent.
    """

    def __init__(
        self,
        max_revisions: int = 2,
        seed: int = 42,
        llm_backend=None,
    ) -> None:
        super().__init__(seed=seed, llm_backend=llm_backend)
        self._strategy_name = "critique_revise"
        self._max_revisions = max_revisions

    def generate_next_turn(self, state: DialogueState) -> str:
        draft = self._pick_question(state)
        norms = state.scenario.recommended_norms

        for revision_num in range(self._max_revisions):
            issues = _critique(draft, norms)
            if not issues:
                logger.debug("CritiqueRevise: draft passed after %d revision(s).", revision_num)
                break
            logger.debug("CritiqueRevise: revision %d — issues: %s", revision_num + 1, issues)

            if self._llm_backend:
                revised = self._call_llm(self._build_revision_prompt(draft, issues, state))
                if revised:
                    draft = revised
                    continue

            draft = _apply_rewrites(draft)

        return draft

    def _build_revision_prompt(
        self, draft: str, issues: list[str], state: DialogueState
    ) -> str:
        issues_str = "\n".join(f"- {i}" for i in issues)
        norms_str = ", ".join(state.scenario.recommended_norms)
        return (
            f"You are revising a dialogue agent's utterance to comply with norms: {norms_str}.\n\n"
            f"Original draft:\n{draft}\n\n"
            f"Issues identified:\n{issues_str}\n\n"
            f"Rewrite the utterance to resolve these issues while preserving the intent. "
            f"Return only the revised utterance, no explanation."
        )
