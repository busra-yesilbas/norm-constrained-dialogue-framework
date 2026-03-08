"""
Pydantic schemas for all core data structures in the framework.

These models ensure type safety, validation, and clean serialisation
throughout the pipeline (generation → simulation → evaluation → reporting).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ScenarioType(str, Enum):
    WORKPLACE_INCIDENT = "workplace_incident_review"
    PUBLIC_SERVICE = "public_service_complaint"
    CLINICAL_INTAKE = "clinical_intake"
    COMPLIANCE_INTERVIEW = "compliance_interview"
    WITNESS_RECALL = "witness_recall"


class RespondentProfile(str, Enum):
    COOPERATIVE = "cooperative"
    ANXIOUS = "anxious"
    RESISTANT = "resistant"
    CONFUSED = "confused"
    FATIGUED = "fatigued"
    TRAUMA_SENSITIVE = "trauma_sensitive"


class SensitivityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentStrategy(str, Enum):
    BASELINE = "baseline"
    RULE_AUGMENTED = "rule_augmented"
    CRITIQUE_REVISE = "critique_revise"
    CANDIDATE_SELECTOR = "candidate_selector"
    CONSTRAINED_FILTER = "constrained_filter"


class Speaker(str, Enum):
    AGENT = "agent"
    RESPONDENT = "respondent"
    SYSTEM = "system"


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


class Scenario(BaseModel):
    """A fully specified synthetic conversation scenario."""

    case_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    scenario_type: ScenarioType
    respondent_profile: RespondentProfile
    context_summary: str
    latent_truth_state: dict[str, str] = Field(
        default_factory=dict,
        description="Hidden ground truth known only to the simulator, not the agent.",
    )
    sensitivity_level: SensitivityLevel
    communication_goals: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    recommended_norms: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"use_enum_values": True}


# ---------------------------------------------------------------------------
# Dialogue turn
# ---------------------------------------------------------------------------


class TurnMetrics(BaseModel):
    """Per-turn evaluation scores for the agent's utterance."""

    coercion_risk: float = Field(0.0, ge=0.0, le=1.0)
    leading_question_risk: float = Field(0.0, ge=0.0, le=1.0)
    pressure_escalation_risk: float = Field(0.0, ge=0.0, le=1.0)
    empathy_score: float = Field(0.5, ge=0.0, le=1.0)
    transparency_score: float = Field(0.5, ge=0.0, le=1.0)
    neutrality_score: float = Field(0.5, ge=0.0, le=1.0)
    procedural_fairness_score: float = Field(0.5, ge=0.0, le=1.0)
    information_yield: float = Field(0.5, ge=0.0, le=1.0)
    trust_preservation: float = Field(0.5, ge=0.0, le=1.0)
    engagement_score: float = Field(0.5, ge=0.0, le=1.0)

    # Aggregate scores (computed by evaluator)
    ethical_alignment_score: float = Field(0.5, ge=0.0, le=1.0)
    utility_score: float = Field(0.5, ge=0.0, le=1.0)
    composite_score: float = Field(0.5, ge=0.0, le=1.0)


class DialogueTurn(BaseModel):
    """A single turn in the conversation."""

    turn_id: int
    speaker: Speaker
    utterance: str
    turn_metrics: Optional[TurnMetrics] = None
    respondent_trust_level: Optional[float] = None
    respondent_stress_level: Optional[float] = None
    metadata: dict = Field(default_factory=dict)

    model_config = {"use_enum_values": True}


# ---------------------------------------------------------------------------
# Dialogue episode
# ---------------------------------------------------------------------------


class DialogueEpisode(BaseModel):
    """A complete conversation episode with all turns and metadata."""

    episode_id: str = Field(default_factory=lambda: str(uuid4())[:12])
    scenario: Scenario
    agent_strategy: AgentStrategy
    turns: list[DialogueTurn] = Field(default_factory=list)
    final_trust_level: Optional[float] = None
    final_stress_level: Optional[float] = None
    total_turns: int = 0
    completed: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"use_enum_values": True}

    def agent_turns(self) -> list[DialogueTurn]:
        return [t for t in self.turns if t.speaker == Speaker.AGENT]

    def respondent_turns(self) -> list[DialogueTurn]:
        return [t for t in self.turns if t.speaker == Speaker.RESPONDENT]


# ---------------------------------------------------------------------------
# Episode metrics summary
# ---------------------------------------------------------------------------


class EpisodeSummary(BaseModel):
    """Aggregated metrics for a completed dialogue episode."""

    episode_id: str
    case_id: str
    scenario_type: str
    respondent_profile: str
    sensitivity_level: str
    agent_strategy: str
    total_turns: int

    # Mean per-turn metrics
    mean_coercion_risk: float
    mean_leading_question_risk: float
    mean_pressure_escalation_risk: float
    mean_empathy_score: float
    mean_transparency_score: float
    mean_neutrality_score: float
    mean_procedural_fairness_score: float
    mean_information_yield: float
    mean_trust_preservation: float
    mean_engagement_score: float

    # Aggregate scores
    ethical_alignment_score: float
    utility_score: float
    composite_score: float

    final_trust_level: float
    final_stress_level: float
    completed: bool


# ---------------------------------------------------------------------------
# Experiment results
# ---------------------------------------------------------------------------


class ExperimentResult(BaseModel):
    """Collects all episode summaries for one experimental run."""

    experiment_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    config_path: str = "configs/default.yaml"
    n_episodes: int = 0
    summaries: list[EpisodeSummary] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
