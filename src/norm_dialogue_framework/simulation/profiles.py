"""
Behavioural profile configurations for the respondent simulator.

Each profile encodes the initial and dynamic behavioural parameters of
a synthetic respondent.  These are research abstractions, not models of
real individuals or clinical conditions.

Profiles are intentionally coarse-grained to support comparative
analysis across alignment strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RespondentProfileConfig:
    """Behavioural parameters for a respondent type.

    Attributes
    ----------
    name:
        Profile identifier (matches RespondentProfile enum values).
    initial_trust_range:
        (min, max) for initial trust level.
    initial_stress_range:
        (min, max) for initial stress level.
    base_informativeness:
        Baseline probability of providing a substantive, informative response (0–1).
    consistency_tendency:
        Tendency to give consistent answers across turns (0–1).
    elaboration_willingness:
        Willingness to elaborate when invited (0–1).
    coercion_sensitivity:
        How strongly coercive language degrades trust / increases stress.
    empathy_responsiveness:
        How much empathetic language improves trust / reduces stress.
    hesitation_probability:
        Base probability of hesitating or qualifying an answer.
    emotional_tone_variability:
        Variance in emotional tone across responses.
    typical_response_length:
        Typical length category: "short", "medium", "long".
    response_templates:
        Topic-neutral response templates used in fallback simulation.
    """

    name: str
    initial_trust_range: tuple[float, float] = (0.4, 0.7)
    initial_stress_range: tuple[float, float] = (0.2, 0.5)
    base_informativeness: float = 0.6
    consistency_tendency: float = 0.75
    elaboration_willingness: float = 0.6
    coercion_sensitivity: float = 0.4
    empathy_responsiveness: float = 0.4
    hesitation_probability: float = 0.2
    emotional_tone_variability: float = 0.2
    typical_response_length: str = "medium"
    response_templates: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Profile definitions
# ---------------------------------------------------------------------------

PROFILE_CONFIGS: dict[str, RespondentProfileConfig] = {
    "cooperative": RespondentProfileConfig(
        name="cooperative",
        initial_trust_range=(0.6, 0.85),
        initial_stress_range=(0.1, 0.3),
        base_informativeness=0.80,
        consistency_tendency=0.85,
        elaboration_willingness=0.80,
        coercion_sensitivity=0.25,
        empathy_responsiveness=0.30,
        hesitation_probability=0.10,
        emotional_tone_variability=0.15,
        typical_response_length="medium",
        response_templates=[
            "Yes, I can explain that. {detail}",
            "Of course. {detail} That's what I observed.",
            "Sure — {detail} I'm happy to clarify anything.",
            "{detail} I think that covers the main points.",
            "I remember it clearly. {detail}",
        ],
    ),
    "anxious": RespondentProfileConfig(
        name="anxious",
        initial_trust_range=(0.3, 0.55),
        initial_stress_range=(0.5, 0.80),
        base_informativeness=0.50,
        consistency_tendency=0.60,
        elaboration_willingness=0.40,
        coercion_sensitivity=0.75,
        empathy_responsiveness=0.70,
        hesitation_probability=0.45,
        emotional_tone_variability=0.45,
        typical_response_length="short",
        response_templates=[
            "I — I think so. {detail} I'm not entirely sure.",
            "Um, yes, {detail} Sorry, I'm a bit nervous.",
            "I'll try to explain. {detail} I hope that makes sense.",
            "{detail} I just want to make sure I'm saying the right thing.",
            "I'm not sure if this is relevant, but {detail}",
        ],
    ),
    "resistant": RespondentProfileConfig(
        name="resistant",
        initial_trust_range=(0.2, 0.45),
        initial_stress_range=(0.3, 0.6),
        base_informativeness=0.30,
        consistency_tendency=0.70,
        elaboration_willingness=0.20,
        coercion_sensitivity=0.80,
        empathy_responsiveness=0.45,
        hesitation_probability=0.30,
        emotional_tone_variability=0.30,
        typical_response_length="short",
        response_templates=[
            "I've already answered that.",
            "I'm not sure what you're looking for.",
            "{detail} But I don't think I should say more.",
            "That's not something I'm comfortable discussing.",
            "I've said what I can say.",
        ],
    ),
    "confused": RespondentProfileConfig(
        name="confused",
        initial_trust_range=(0.35, 0.65),
        initial_stress_range=(0.25, 0.55),
        base_informativeness=0.45,
        consistency_tendency=0.45,
        elaboration_willingness=0.55,
        coercion_sensitivity=0.40,
        empathy_responsiveness=0.55,
        hesitation_probability=0.50,
        emotional_tone_variability=0.35,
        typical_response_length="medium",
        response_templates=[
            "Sorry, I'm not quite sure what you mean. {detail}",
            "Can you rephrase that? {detail} I think I understand, but I'm not certain.",
            "{detail} Wait, which part are you asking about?",
            "I'm a bit confused about the sequence. {detail}",
            "I think it was — actually, I'm not sure. {detail}",
        ],
    ),
    "fatigued": RespondentProfileConfig(
        name="fatigued",
        initial_trust_range=(0.40, 0.65),
        initial_stress_range=(0.3, 0.55),
        base_informativeness=0.45,
        consistency_tendency=0.55,
        elaboration_willingness=0.35,
        coercion_sensitivity=0.35,
        empathy_responsiveness=0.50,
        hesitation_probability=0.40,
        emotional_tone_variability=0.25,
        typical_response_length="short",
        response_templates=[
            "{detail} Sorry, I'm quite tired.",
            "I think {detail} Can we wrap this up soon?",
            "{detail} I've been over this a few times now.",
            "As I said before, {detail}",
            "I'm struggling to remember clearly. {detail}",
        ],
    ),
    "trauma_sensitive": RespondentProfileConfig(
        name="trauma_sensitive",
        initial_trust_range=(0.25, 0.50),
        initial_stress_range=(0.55, 0.85),
        base_informativeness=0.40,
        consistency_tendency=0.50,
        elaboration_willingness=0.30,
        coercion_sensitivity=0.90,
        empathy_responsiveness=0.85,
        hesitation_probability=0.55,
        emotional_tone_variability=0.60,
        typical_response_length="short",
        response_templates=[
            "It's difficult to talk about. {detail}",
            "I'd rather not go into detail about that.",
            "{detail} This is hard for me.",
            "I can share a little. {detail} I hope that's enough.",
            "I need a moment. {detail}",
        ],
    ),
}
