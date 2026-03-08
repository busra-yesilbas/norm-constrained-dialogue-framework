"""
Dialogue agent strategies for the Norm-Constrained Dialogue Framework.

Each agent implements a common interface:
    generate_next_turn(dialogue_state: DialogueState) -> str

All agents operate in fallback template mode by default (no external
LLM required).  Optionally they can delegate to an OpenAI-compatible
backend when LLM_BACKEND is configured.
"""

from norm_dialogue_framework.agents.base_agent import BaseAgent, DialogueState
from norm_dialogue_framework.agents.baseline_agent import BaselineAgent
from norm_dialogue_framework.agents.rule_augmented_agent import RuleAugmentedAgent
from norm_dialogue_framework.agents.critique_revise_agent import CritiqueReviseAgent
from norm_dialogue_framework.agents.candidate_selector_agent import CandidateSelectorAgent
from norm_dialogue_framework.agents.constrained_filter import ConstrainedFilterAgent

__all__ = [
    "BaseAgent",
    "DialogueState",
    "BaselineAgent",
    "RuleAugmentedAgent",
    "CritiqueReviseAgent",
    "CandidateSelectorAgent",
    "ConstrainedFilterAgent",
]
