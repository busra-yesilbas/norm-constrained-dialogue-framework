"""
Tests for the synthetic scenario generator.
"""

from __future__ import annotations

import pytest

from norm_dialogue_framework.data.synthetic_generator import SyntheticScenarioGenerator
from norm_dialogue_framework.schemas import (
    RespondentProfile,
    ScenarioType,
    Scenario,
    SensitivityLevel,
)


class TestSyntheticScenarioGenerator:
    """Tests for SyntheticScenarioGenerator."""

    def setup_method(self):
        self.gen = SyntheticScenarioGenerator(seed=42)

    def test_generate_returns_correct_count(self):
        scenarios = self.gen.generate(n=10)
        assert len(scenarios) == 10

    def test_generate_single_scenario(self):
        scenarios = self.gen.generate(n=1)
        assert len(scenarios) == 1
        assert isinstance(scenarios[0], Scenario)

    def test_scenario_has_valid_schema(self):
        scenario = self.gen.generate(n=1)[0]
        assert scenario.case_id
        assert scenario.scenario_type in [e.value for e in ScenarioType]
        assert scenario.respondent_profile in [e.value for e in RespondentProfile]
        assert scenario.sensitivity_level in [e.value for e in SensitivityLevel]
        assert isinstance(scenario.context_summary, str)
        assert len(scenario.context_summary) > 10
        assert isinstance(scenario.communication_goals, list)
        assert len(scenario.communication_goals) > 0
        assert isinstance(scenario.risk_flags, list)
        assert isinstance(scenario.recommended_norms, list)

    def test_scenarios_are_unique(self):
        scenarios = self.gen.generate(n=20)
        case_ids = [s.case_id for s in scenarios]
        # UUIDs should be unique
        assert len(set(case_ids)) == len(case_ids)

    def test_reproducibility_with_same_seed(self):
        gen1 = SyntheticScenarioGenerator(seed=99)
        gen2 = SyntheticScenarioGenerator(seed=99)
        s1 = gen1.generate(n=5)
        s2 = gen2.generate(n=5)
        # Context summaries should be identical (case_ids use uuid4 so differ)
        for a, b in zip(s1, s2):
            assert a.scenario_type == b.scenario_type
            assert a.respondent_profile == b.respondent_profile
            assert a.sensitivity_level == b.sensitivity_level

    def test_filter_by_scenario_type(self):
        gen = SyntheticScenarioGenerator(
            seed=42, scenario_types=["workplace_incident_review"]
        )
        scenarios = gen.generate(n=10)
        assert all(s.scenario_type == "workplace_incident_review" for s in scenarios)

    def test_filter_by_profile(self):
        gen = SyntheticScenarioGenerator(
            seed=42, respondent_profiles=["anxious"]
        )
        scenarios = gen.generate(n=10)
        assert all(s.respondent_profile == "anxious" for s in scenarios)

    def test_to_dataframe(self):
        scenarios = self.gen.generate(n=5)
        df = self.gen.to_dataframe(scenarios)
        assert len(df) == 5
        required_cols = ["case_id", "scenario_type", "respondent_profile", "sensitivity_level", "context_summary"]
        for col in required_cols:
            assert col in df.columns

    def test_large_generation(self):
        scenarios = self.gen.generate(n=100)
        assert len(scenarios) == 100
        # Check all scenario types are represented
        types = {s.scenario_type for s in scenarios}
        assert len(types) > 1
        # Check all profiles are represented
        profiles = {s.respondent_profile for s in scenarios}
        assert len(profiles) > 1

    def test_latent_truth_state_is_dict(self):
        scenario = self.gen.generate(n=1)[0]
        assert isinstance(scenario.latent_truth_state, dict)

    def test_no_real_institution_references(self):
        """Ensure generated text does not reference real institutions (smoke test)."""
        scenarios = self.gen.generate(n=20)
        forbidden = ["police", "FBI", "hospital", "court", "government", "NHS"]
        for s in scenarios:
            text = s.context_summary.lower()
            for term in forbidden:
                assert term.lower() not in text, (
                    f"Real institution reference '{term}' found in scenario {s.case_id}"
                )
