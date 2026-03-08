"""
Abstract base class for all dialogue agent strategies.

Defines the common interface and shared utilities (fallback template
generation, optional LLM delegation) used by all concrete agents.
"""

from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from norm_dialogue_framework.schemas import DialogueTurn, Scenario
from norm_dialogue_framework.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dialogue state container
# ---------------------------------------------------------------------------


@dataclass
class DialogueState:
    """Encapsulates the current state of an ongoing conversation.

    Attributes
    ----------
    scenario:
        The scenario driving this conversation.
    history:
        All turns so far (agent + respondent), in order.
    trust_level:
        Current trust level of the respondent (0–1).
    stress_level:
        Current stress level of the respondent (0–1).
    turn_number:
        Current turn index (0-based).
    goals_addressed:
        Subset of communication goals that have been addressed so far.
    """

    scenario: Scenario
    history: list[DialogueTurn] = field(default_factory=list)
    trust_level: float = 0.5
    stress_level: float = 0.3
    turn_number: int = 0
    goals_addressed: list[str] = field(default_factory=list)

    @property
    def last_respondent_utterance(self) -> Optional[str]:
        for t in reversed(self.history):
            if t.speaker == "respondent":
                return t.utterance
        return None

    @property
    def last_agent_utterance(self) -> Optional[str]:
        for t in reversed(self.history):
            if t.speaker == "agent":
                return t.utterance
        return None

    @property
    def n_agent_turns(self) -> int:
        return sum(1 for t in self.history if t.speaker == "agent")


# ---------------------------------------------------------------------------
# Template question banks (fallback mode)
# ---------------------------------------------------------------------------

_OPENING_QUESTIONS: list[str] = [
    "Thank you for taking the time to speak with me today. Could you start by describing what happened, in your own words?",
    "I appreciate you being here. To begin, could you walk me through what you observed or experienced?",
    "Before we start, I want you to know you can take your time. Could you tell me what you remember about the situation?",
    "Thank you for joining this conversation. In your own words, could you describe the events as you understand them?",
]

_FOLLOWUP_QUESTIONS: list[str] = [
    "Could you tell me a bit more about that?",
    "I'd like to understand that better — what happened next?",
    "When you mention {topic}, could you elaborate on what you mean?",
    "You mentioned {topic}. Was there anything else you noticed at that point?",
    "That's helpful. Could you describe what you saw or heard just before that?",
    "Is there any other detail about that part that stands out to you?",
    "How were you feeling during that moment?",
    "Was there anyone else present when that occurred?",
    "Can you clarify what you mean by that — in your own words?",
    "What was your immediate reaction at that point?",
]

_CLARIFICATION_QUESTIONS: list[str] = [
    "Just to make sure I understand correctly — are you saying that {paraphrase}?",
    "Could you help me understand the timing a bit better?",
    "When you say that, do you mean it happened before or after {event}?",
    "I want to make sure I've captured this accurately — would you mind describing that once more?",
]

_CLOSING_QUESTIONS: list[str] = [
    "Is there anything else you think is important for me to understand about this situation?",
    "We're nearly done. Is there anything you'd like to add or clarify before we finish?",
    "Thank you so much for sharing this. Is there anything else on your mind that you haven't had a chance to mention?",
]


def _extract_topic(utterance: str) -> str:
    """Extract a simple topic word from a respondent utterance (heuristic)."""
    words = utterance.split()
    # Pick the first long non-stopword
    stopwords = {
        "i",
        "the",
        "a",
        "was",
        "it",
        "that",
        "and",
        "to",
        "in",
        "of",
        "my",
        "me",
        "is",
        "on",
        "at",
        "this",
    }
    for w in words:
        cleaned = w.strip(".,!?\"'").lower()
        if len(cleaned) > 3 and cleaned not in stopwords:
            return cleaned
    return "that"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """Abstract base class for all dialogue agent strategies.

    Subclasses must implement `generate_next_turn`.

    Parameters
    ----------
    seed:
        Random seed for reproducibility.
    llm_backend:
        Optional LLM backend identifier.  When ``None`` (default),
        all agents run in template-based fallback mode.
    """

    def __init__(self, seed: int = 42, llm_backend: Optional[str] = None) -> None:
        self._rng = random.Random(seed)
        self._llm_backend = llm_backend or os.getenv("LLM_BACKEND") or None
        self._strategy_name: str = "base"

    @property
    def strategy_name(self) -> str:
        return self._strategy_name

    @abstractmethod
    def generate_next_turn(self, state: DialogueState) -> str:
        """Generate the agent's next utterance given the current dialogue state.

        Parameters
        ----------
        state:
            Current dialogue state including history, scenario, and
            respondent metrics.

        Returns
        -------
        str
            The agent's next utterance text.
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _pick_question(self, state: DialogueState) -> str:
        """Select an appropriate template question based on turn number."""
        n = state.n_agent_turns
        if n == 0:
            return self._rng.choice(_OPENING_QUESTIONS)

        last_resp = state.last_respondent_utterance or ""

        # Final turns
        remaining = (state.scenario.sensitivity_level == "critical" and n >= 8) or n >= 10
        if remaining:
            return self._rng.choice(_CLOSING_QUESTIONS)

        # Occasionally use clarification
        if last_resp and self._rng.random() < 0.25:
            q = self._rng.choice(_CLARIFICATION_QUESTIONS)
            return q.replace("{paraphrase}", last_resp[:60]).replace(
                "{event}", _extract_topic(last_resp)
            )

        q = self._rng.choice(_FOLLOWUP_QUESTIONS)
        return q.replace("{topic}", _extract_topic(last_resp))

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Attempt to call a configured LLM backend.

        Returns ``None`` if no backend is configured or if the call fails,
        triggering fallback to template generation.
        """
        if not self._llm_backend:
            return None
        try:
            if self._llm_backend == "openai":
                return self._call_openai(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM call failed (%s); falling back to template mode.", exc)
        return None

    def _call_openai(self, prompt: str) -> str:
        """Call the OpenAI-compatible API."""
        import openai  # type: ignore

        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
