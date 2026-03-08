"""
Respondent simulator for the Norm-Constrained Dialogue Framework.

Models a synthetic respondent whose behaviour evolves dynamically
based on their behavioural profile, the current trust and stress levels,
and the perceived quality of the agent's utterances.

This is a research abstraction.  It does not model real individuals,
clinical conditions, or any specific population.
"""

from __future__ import annotations

import random
from typing import Optional

from norm_dialogue_framework.schemas import Scenario
from norm_dialogue_framework.simulation.profiles import PROFILE_CONFIGS, RespondentProfileConfig
from norm_dialogue_framework.utils import clamp, get_logger, sample_uniform

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Detail fragments for response assembly
# ---------------------------------------------------------------------------

_DETAIL_FRAGMENTS: list[str] = [
    "I was in the area at the time and noticed what was happening.",
    "I remember seeing someone near the entrance when it started.",
    "The situation felt unusual, and I tried to stay calm.",
    "I documented what I could at the time, though it was hard.",
    "I wasn't directly involved, but I saw part of what happened.",
    "It was a stressful situation for everyone present.",
    "I had already flagged a concern beforehand, but nothing changed.",
    "From where I was standing, I had a partial view of events.",
    "I spoke to someone briefly before this conversation, but they didn't say much.",
    "I'm still processing some of what happened, to be honest.",
    "There was a lot going on at once, so my memory of details is a bit blurry.",
    "At the time, I wasn't sure whether to step in or stay back.",
    "I followed standard procedure as best I could given the circumstances.",
    "The sequence of events happened quickly, so I may have missed some details.",
]

_HESITATION_PREFIXES: list[str] = [
    "Well... ",
    "Hmm, let me think... ",
    "That's a good question... ",
    "I'm not entirely sure, but... ",
    "Give me a moment... ",
    "I'll try to explain this clearly... ",
]

_EMOTIONAL_QUALIFIERS: dict[str, list[str]] = {
    "anxious": [
        "I hope this is helpful.",
        "I'm sorry if I'm not being clear.",
        "I'm a bit overwhelmed.",
        "Please bear with me.",
    ],
    "resistant": [
        "I've already explained this.",
        "I'm not sure what more you need.",
        "I'd prefer to move on from this.",
    ],
    "confused": [
        "I think that's right, but I'm not certain.",
        "Let me try to clarify.",
        "I may be misremembering.",
    ],
    "fatigued": [
        "I'm quite tired.",
        "I've been through this before.",
        "Can we keep this brief?",
    ],
    "trauma_sensitive": [
        "This is difficult to discuss.",
        "I'm trying my best.",
        "I appreciate your patience.",
    ],
    "cooperative": [
        "Happy to help.",
        "Let me know if you need more detail.",
        "I'll try to be as clear as possible.",
    ],
}


class RespondentSimulator:
    """Simulates a synthetic respondent in a structured dialogue.

    Maintains internal trust, stress, and fatigue state that evolves
    across turns in response to the agent's utterances.

    Parameters
    ----------
    scenario:
        The scenario defining context and respondent profile.
    seed:
        Random seed for reproducibility.
    initial_trust:
        Override for initial trust level (0–1).  If None, sampled from
        the profile's ``initial_trust_range``.
    initial_stress:
        Override for initial stress level (0–1).  If None, sampled
        from the profile's ``initial_stress_range``.
    """

    def __init__(
        self,
        scenario: Scenario,
        seed: int = 42,
        initial_trust: Optional[float] = None,
        initial_stress: Optional[float] = None,
    ) -> None:
        self._scenario = scenario
        self._rng = random.Random(seed)
        self._profile_name: str = scenario.respondent_profile  # type: ignore[assignment]
        self._profile: RespondentProfileConfig = PROFILE_CONFIGS.get(
            self._profile_name, PROFILE_CONFIGS["cooperative"]
        )

        lo_t, hi_t = self._profile.initial_trust_range
        lo_s, hi_s = self._profile.initial_stress_range

        self.trust_level: float = clamp(
            initial_trust if initial_trust is not None else self._rng.uniform(lo_t, hi_t)
        )
        self.stress_level: float = clamp(
            initial_stress if initial_stress is not None else self._rng.uniform(lo_s, hi_s)
        )
        self._turn_count: int = 0
        self._fatigue: float = 0.0
        logger.debug(
            "Respondent initialised: profile=%s trust=%.2f stress=%.2f",
            self._profile_name,
            self.trust_level,
            self.stress_level,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def respond(self, agent_utterance: str) -> str:
        """Generate the respondent's response to an agent utterance.

        Parameters
        ----------
        agent_utterance:
            The agent's latest utterance.

        Returns
        -------
        str
            The respondent's response text.
        """
        self._update_internal_state(agent_utterance)
        response = self._generate_response(agent_utterance)
        self._turn_count += 1
        self._fatigue = clamp(self._fatigue + 0.04)
        return response

    @property
    def profile_name(self) -> str:
        return self._profile_name

    # ------------------------------------------------------------------
    # Internal dynamics
    # ------------------------------------------------------------------

    def _update_internal_state(self, agent_utterance: str) -> None:
        """Update trust and stress based on the agent's utterance."""
        u = agent_utterance.lower()

        # Coercion detection — decreases trust, increases stress
        coercive_signals = [
            "you must",
            "you have to",
            "admit",
            "confess",
            "don't lie",
            "we know",
        ]
        coercion_hit = any(s in u for s in coercive_signals)
        if coercion_hit:
            delta = self._profile.coercion_sensitivity * self._rng.uniform(0.1, 0.3)
            self.trust_level = clamp(self.trust_level - delta)
            self.stress_level = clamp(self.stress_level + delta * 0.8)
            logger.debug("Respondent: coercion detected → trust−%.2f stress+%.2f", delta, delta * 0.8)

        # Empathy detection — increases trust, decreases stress
        empathy_signals = [
            "i understand",
            "that must be",
            "thank you",
            "i appreciate",
            "take your time",
            "in your own words",
        ]
        empathy_hit = any(s in u for s in empathy_signals)
        if empathy_hit:
            delta = self._profile.empathy_responsiveness * self._rng.uniform(0.05, 0.15)
            self.trust_level = clamp(self.trust_level + delta)
            self.stress_level = clamp(self.stress_level - delta * 0.7)
            logger.debug("Respondent: empathy detected → trust+%.2f stress−%.2f", delta, delta * 0.7)

        # Natural drift: trust erodes slightly over long conversations
        if self._turn_count > 6:
            self.trust_level = clamp(self.trust_level - 0.01)

        # Stress increases slightly with fatigue
        self.stress_level = clamp(self.stress_level + self._fatigue * 0.02)

    def _generate_response(self, agent_utterance: str) -> str:
        """Assemble a contextually appropriate response."""
        informativeness = self._effective_informativeness()
        is_informative = self._rng.random() < informativeness
        detail = self._rng.choice(_DETAIL_FRAGMENTS) if is_informative else "I'm not sure I can say more about that."

        # Pick a template from the profile
        templates = self._profile.response_templates
        if templates:
            template = self._rng.choice(templates)
            response = template.replace("{detail}", detail)
        else:
            response = detail

        # Optionally add hesitation
        if self._rng.random() < self._effective_hesitation():
            prefix = self._rng.choice(_HESITATION_PREFIXES)
            response = prefix + response

        # Optionally add emotional qualifier
        qualifiers = _EMOTIONAL_QUALIFIERS.get(self._profile_name, [])
        if qualifiers and self._rng.random() < self._profile.emotional_tone_variability:
            response = response + " " + self._rng.choice(qualifiers)

        return response.strip()

    def _effective_informativeness(self) -> float:
        """Informativeness modulated by trust, stress, and fatigue."""
        base = self._profile.base_informativeness
        trust_boost = (self.trust_level - 0.5) * 0.3
        stress_penalty = self.stress_level * 0.2
        fatigue_penalty = self._fatigue * 0.15
        return clamp(base + trust_boost - stress_penalty - fatigue_penalty)

    def _effective_hesitation(self) -> float:
        """Hesitation probability modulated by stress and fatigue."""
        base = self._profile.hesitation_probability
        stress_boost = self.stress_level * 0.2
        return clamp(base + stress_boost)
