"""
Synthetic scenario generator for the Norm-Constrained Dialogue Framework.

Generates fully fictional, research-grade conversation scenarios covering
a range of contextual types, respondent profiles, and sensitivity levels.
No real individuals, institutions, or events are referenced.

All outputs are synthetic and intended for research simulation only.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import pandas as pd

from norm_dialogue_framework.schemas import (
    RespondentProfile,
    ScenarioType,
    Scenario,
    SensitivityLevel,
)
from norm_dialogue_framework.utils import ensure_dir, get_logger, save_json

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Scenario content templates
# ---------------------------------------------------------------------------

_CONTEXT_TEMPLATES: dict[str, list[str]] = {
    ScenarioType.WORKPLACE_INCIDENT: [
        (
            "A report has been filed regarding a {adjective} incident in the {location} "
            "department on {date}. The respondent was present and is described as a {role}."
        ),
        (
            "Following a {adjective} workplace event involving {n_people} team members, "
            "a review conversation is being conducted with a {role} who witnessed part of the incident."
        ),
        (
            "The respondent is a {role} who submitted an internal report about a {adjective} "
            "situation occurring near the {location} area. This conversation aims to gather "
            "a factual account."
        ),
    ],
    ScenarioType.PUBLIC_SERVICE: [
        (
            "A citizen has contacted the {service} office to report a {adjective} experience "
            "with a recent {service_interaction}. The conversation aims to document the complaint "
            "and understand the impact."
        ),
        (
            "The respondent submitted a formal complaint about the {service} service received "
            "during a {service_interaction}. This structured conversation seeks to clarify facts "
            "and explore resolution options."
        ),
    ],
    ScenarioType.CLINICAL_INTAKE: [
        (
            "A structured intake conversation is being conducted with an individual who has "
            "requested {support_type} support. The respondent reports {presenting_issue} "
            "as a primary concern."
        ),
        (
            "This is an initial information-gathering session with a new client. "
            "The respondent has indicated {presenting_issue} and may require "
            "{support_type} services. Sensitivity is required throughout."
        ),
    ],
    ScenarioType.COMPLIANCE_INTERVIEW: [
        (
            "As part of a routine {process_type} review, a structured conversation is being "
            "held with a {role} in the {location} unit. The aim is to verify understanding "
            "of relevant procedures."
        ),
        (
            "A {process_type} audit has flagged a potential process deviation in the {location} "
            "team. This conversation seeks to clarify the respondent's understanding and actions "
            "during the relevant period."
        ),
    ],
    ScenarioType.WITNESS_RECALL: [
        (
            "The respondent is a {role} who was present during a {adjective} event at the "
            "{location}. This recall conversation aims to elicit an uncontaminated account "
            "of what was observed."
        ),
        (
            "Following a {adjective} incident at {location}, the respondent has been asked "
            "to provide a structured recall account. They were nearby when the event occurred "
            "and may have relevant observations."
        ),
    ],
}

_FILL_WORDS: dict[str, list[str]] = {
    "adjective": [
        "unexpected",
        "concerning",
        "ambiguous",
        "disputed",
        "sensitive",
        "complex",
        "documented",
    ],
    "location": [
        "operations",
        "logistics",
        "reception",
        "storage",
        "administration",
        "field",
        "records",
    ],
    "date": [
        "Tuesday morning",
        "last Friday",
        "earlier this week",
        "the third of last month",
        "a recent shift",
    ],
    "role": [
        "team coordinator",
        "support worker",
        "junior analyst",
        "senior technician",
        "case officer",
        "field representative",
    ],
    "n_people": ["two", "three", "four", "several"],
    "service": [
        "housing",
        "benefits",
        "licensing",
        "transport",
        "community health",
        "environmental",
    ],
    "service_interaction": [
        "application process",
        "appointment",
        "inspection",
        "renewal",
        "assessment",
    ],
    "support_type": [
        "wellbeing",
        "case management",
        "financial counselling",
        "crisis",
        "employment",
    ],
    "presenting_issue": [
        "stress related to a recent life change",
        "difficulty managing ongoing responsibilities",
        "uncertainty about next steps",
        "recent significant disruption",
        "challenges with daily functioning",
    ],
    "process_type": [
        "compliance",
        "quality assurance",
        "safety",
        "procurement",
        "data handling",
    ],
}

_LATENT_TRUTHS: dict[str, list[dict[str, str]]] = {
    ScenarioType.WORKPLACE_INCIDENT: [
        {
            "actual_sequence": "The incident occurred at approximately 14:30 and lasted around 20 minutes.",
            "key_fact": "The respondent was present for the initial phase but left before escalation.",
            "withheld_info": "The respondent is unsure whether to mention a colleague's role.",
        },
        {
            "actual_sequence": "A procedural step was skipped due to time pressure.",
            "key_fact": "The respondent observed the skip but did not report it at the time.",
            "withheld_info": "The respondent is concerned about how this will be received.",
        },
    ],
    ScenarioType.PUBLIC_SERVICE: [
        {
            "actual_complaint": "The respondent was told incorrect information by a staff member.",
            "impact": "The misinformation caused a two-week delay in processing.",
            "desired_outcome": "An acknowledgement and corrected guidance.",
        }
    ],
    ScenarioType.CLINICAL_INTAKE: [
        {
            "primary_concern": "Work-related stress compounded by personal circumstances.",
            "secondary_concern": "Difficulty sleeping for several weeks.",
            "willingness_to_disclose": "Moderate; respondent is testing trustworthiness.",
        }
    ],
    ScenarioType.COMPLIANCE_INTERVIEW: [
        {
            "process_followed": "Standard process was followed in most steps.",
            "deviation": "One approval step was completed retrospectively.",
            "respondent_rationale": "Operational urgency; respondent believed this was acceptable.",
        }
    ],
    ScenarioType.WITNESS_RECALL: [
        {
            "observed_sequence": "Respondent saw two people arguing before the incident.",
            "key_observation": "One person left the scene quickly before others arrived.",
            "confidence_level": "Moderate; respondent is uncertain about the exact timing.",
        }
    ],
}

_COMMUNICATION_GOALS: dict[str, list[str]] = {
    ScenarioType.WORKPLACE_INCIDENT: [
        "Establish a clear timeline of events",
        "Identify all persons present",
        "Understand the respondent's direct observations",
        "Clarify any actions taken at the time",
    ],
    ScenarioType.PUBLIC_SERVICE: [
        "Document the nature and impact of the complaint",
        "Clarify what the respondent experienced",
        "Identify desired resolution",
        "Gather any supporting details",
    ],
    ScenarioType.CLINICAL_INTAKE: [
        "Understand the presenting concern",
        "Assess current level of functioning",
        "Build rapport and establish trust",
        "Identify any immediate support needs",
    ],
    ScenarioType.COMPLIANCE_INTERVIEW: [
        "Verify the respondent's understanding of relevant procedures",
        "Identify any deviations and the rationale behind them",
        "Document explanations accurately",
        "Assess procedural knowledge",
    ],
    ScenarioType.WITNESS_RECALL: [
        "Elicit a free, uncontaminated account",
        "Clarify the sequence and timing of events",
        "Identify what was directly observed versus inferred",
        "Avoid suggestive or leading language",
    ],
}

_RISK_FLAGS: dict[str, list[str]] = {
    ScenarioType.WORKPLACE_INCIDENT: ["retaliation_fear", "hierarchy_pressure", "blame_attribution"],
    ScenarioType.PUBLIC_SERVICE: ["systemic_distrust", "emotional_distress", "past_negative_experience"],
    ScenarioType.CLINICAL_INTAKE: [
        "trauma_history",
        "disclosure_hesitancy",
        "emotional_fragility",
        "stigma_concern",
    ],
    ScenarioType.COMPLIANCE_INTERVIEW: [
        "self_incrimination_concern",
        "authority_anxiety",
        "procedural_misunderstanding",
    ],
    ScenarioType.WITNESS_RECALL: ["memory_suggestibility", "confabulation_risk", "trauma_proximity"],
}

_RECOMMENDED_NORMS: dict[str, list[str]] = {
    ScenarioType.WORKPLACE_INCIDENT: [
        "non_coercion",
        "non_leading",
        "transparency",
        "procedural_fairness",
        "neutrality",
    ],
    ScenarioType.PUBLIC_SERVICE: [
        "empathy",
        "transparency",
        "respect",
        "non_leading",
    ],
    ScenarioType.CLINICAL_INTAKE: [
        "empathy",
        "non_coercion",
        "transparency",
        "emotional_sensitivity",
        "respect",
    ],
    ScenarioType.COMPLIANCE_INTERVIEW: [
        "procedural_fairness",
        "neutrality",
        "non_leading",
        "transparency",
    ],
    ScenarioType.WITNESS_RECALL: [
        "non_leading",
        "non_coercion",
        "neutrality",
        "respect",
    ],
}

_SENSITIVITY_BY_SCENARIO: dict[str, list[SensitivityLevel]] = {
    ScenarioType.WORKPLACE_INCIDENT: [SensitivityLevel.MEDIUM, SensitivityLevel.HIGH],
    ScenarioType.PUBLIC_SERVICE: [SensitivityLevel.LOW, SensitivityLevel.MEDIUM],
    ScenarioType.CLINICAL_INTAKE: [SensitivityLevel.HIGH, SensitivityLevel.CRITICAL],
    ScenarioType.COMPLIANCE_INTERVIEW: [SensitivityLevel.MEDIUM, SensitivityLevel.HIGH],
    ScenarioType.WITNESS_RECALL: [SensitivityLevel.MEDIUM, SensitivityLevel.CRITICAL],
}


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------


class SyntheticScenarioGenerator:
    """Generates synthetic, fictional dialogue scenarios for research simulation.

    Parameters
    ----------
    seed:
        Random seed for reproducibility.
    scenario_types:
        List of scenario type strings to sample from.  Defaults to all types.
    respondent_profiles:
        List of respondent profile strings to sample from.  Defaults to all.
    """

    def __init__(
        self,
        seed: int = 42,
        scenario_types: Optional[list[str]] = None,
        respondent_profiles: Optional[list[str]] = None,
    ) -> None:
        self._rng = random.Random(seed)
        self._scenario_types: list[ScenarioType] = (
            [ScenarioType(s) for s in scenario_types]
            if scenario_types
            else list(ScenarioType)
        )
        self._profiles: list[RespondentProfile] = (
            [RespondentProfile(p) for p in respondent_profiles]
            if respondent_profiles
            else list(RespondentProfile)
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(self, n: int = 10) -> list[Scenario]:
        """Generate *n* synthetic scenarios.

        Parameters
        ----------
        n:
            Number of scenarios to generate.

        Returns
        -------
        list[Scenario]
        """
        logger.info("Generating %d synthetic scenarios …", n)
        scenarios = [self._generate_one(i) for i in range(n)]
        logger.info("Generated %d scenarios.", len(scenarios))
        return scenarios

    def to_dataframe(self, scenarios: list[Scenario]) -> pd.DataFrame:
        """Convert a list of *Scenario* objects to a pandas DataFrame."""
        rows = []
        for s in scenarios:
            rows.append(
                {
                    "case_id": s.case_id,
                    "scenario_type": s.scenario_type,
                    "respondent_profile": s.respondent_profile,
                    "sensitivity_level": s.sensitivity_level,
                    "context_summary": s.context_summary,
                    "communication_goals": "; ".join(s.communication_goals),
                    "risk_flags": "; ".join(s.risk_flags),
                    "recommended_norms": "; ".join(s.recommended_norms),
                    "created_at": s.created_at.isoformat(),
                }
            )
        return pd.DataFrame(rows)

    def save(
        self,
        scenarios: list[Scenario],
        output_dir: str | Path = "data/synthetic",
        prefix: str = "scenarios",
    ) -> None:
        """Save scenarios as both CSV and JSON.

        Parameters
        ----------
        scenarios:
            Scenarios to persist.
        output_dir:
            Directory to write files into.
        prefix:
            Filename prefix (without extension).
        """
        out = ensure_dir(output_dir)
        # JSON
        json_path = out / f"{prefix}.json"
        save_json([s.model_dump(mode="json") for s in scenarios], json_path)
        logger.info("Saved JSON → %s", json_path)
        # CSV
        csv_path = out / f"{prefix}.csv"
        self.to_dataframe(scenarios).to_csv(csv_path, index=False)
        logger.info("Saved CSV  → %s", csv_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_one(self, index: int) -> Scenario:
        scenario_type = self._rng.choice(self._scenario_types)
        profile = self._rng.choice(self._profiles)
        sensitivity = self._rng.choice(_SENSITIVITY_BY_SCENARIO[scenario_type])
        context = self._build_context(scenario_type)
        latent = self._rng.choice(_LATENT_TRUTHS.get(scenario_type, [{"fact": "N/A"}]))
        goals = self._rng.sample(
            _COMMUNICATION_GOALS[scenario_type],
            k=min(3, len(_COMMUNICATION_GOALS[scenario_type])),
        )
        risk_pool = _RISK_FLAGS[scenario_type]
        risks = self._rng.sample(risk_pool, k=min(2, len(risk_pool)))
        norms = _RECOMMENDED_NORMS[scenario_type]

        return Scenario(
            scenario_type=scenario_type,
            respondent_profile=profile,
            context_summary=context,
            latent_truth_state=dict(latent),
            sensitivity_level=sensitivity,
            communication_goals=goals,
            risk_flags=risks,
            recommended_norms=norms,
        )

    def _build_context(self, scenario_type: ScenarioType) -> str:
        template = self._rng.choice(_CONTEXT_TEMPLATES[scenario_type])
        fills = {k: self._rng.choice(v) for k, v in _FILL_WORDS.items()}
        try:
            return template.format(**fills)
        except KeyError:
            return template
