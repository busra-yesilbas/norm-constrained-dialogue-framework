"""
Baseline dialogue agent — no norm constraints applied.

Selects questions using simple template heuristics without any
explicit norm checking, critique, or candidate scoring.  Serves as
the control condition in strategy comparison experiments.
"""

from __future__ import annotations

from norm_dialogue_framework.agents.base_agent import BaseAgent, DialogueState
from norm_dialogue_framework.utils import get_logger

logger = get_logger(__name__)


class BaselineAgent(BaseAgent):
    """Baseline agent with no explicit norm guidance.

    Generates questions using template heuristics or (optionally) a
    raw LLM call with a minimal system prompt.  No rule checking,
    critique loop, or candidate scoring is applied.

    This agent intentionally represents a weaker alignment baseline so
    that more constrained strategies can be meaningfully compared.
    """

    def __init__(self, seed: int = 42, llm_backend=None) -> None:
        super().__init__(seed=seed, llm_backend=llm_backend)
        self._strategy_name = "baseline"

    def generate_next_turn(self, state: DialogueState) -> str:
        """Generate the next agent utterance with no norm constraints.

        Parameters
        ----------
        state:
            Current dialogue state.

        Returns
        -------
        str
            The agent's next utterance.
        """
        if self._llm_backend:
            prompt = self._build_prompt(state)
            result = self._call_llm(prompt)
            if result:
                return result

        return self._pick_question(state)

    def _build_prompt(self, state: DialogueState) -> str:
        context = state.scenario.context_summary
        history_lines = "\n".join(
            f"{t.speaker.upper()}: {t.utterance}" for t in state.history[-6:]
        )
        return (
            f"You are a conversational agent conducting a structured information-gathering "
            f"conversation.\n\nContext: {context}\n\nRecent dialogue:\n{history_lines}\n\n"
            f"Generate the agent's next question or statement."
        )
