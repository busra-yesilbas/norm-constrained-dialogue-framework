"""
Tests for the evaluation pipeline.
"""

from __future__ import annotations

import pytest

from norm_dialogue_framework.agents.baseline_agent import BaselineAgent
from norm_dialogue_framework.agents.candidate_selector_agent import CandidateSelectorAgent
from norm_dialogue_framework.data.synthetic_generator import SyntheticScenarioGenerator
from norm_dialogue_framework.evaluation.evaluator import Evaluator
from norm_dialogue_framework.evaluation.metrics import MetricsComputer
from norm_dialogue_framework.evaluation.reward_model import RewardModel
from norm_dialogue_framework.evaluation.rule_checks import RuleChecker
from norm_dialogue_framework.schemas import EpisodeSummary, TurnMetrics
from norm_dialogue_framework.simulation.dialogue_runner import DialogueRunner


@pytest.fixture
def scenario():
    gen = SyntheticScenarioGenerator(seed=7)
    return gen.generate(n=1)[0]


@pytest.fixture
def completed_episode(scenario):
    agent = BaselineAgent(seed=7)
    evaluator = Evaluator()
    runner = DialogueRunner(agent=agent, max_turns=5, min_turns=2, seed=7, evaluator=evaluator)
    return runner.run(scenario)


class TestEvaluator:
    """Tests for the Evaluator class."""

    def test_evaluate_episode_returns_summary(self, completed_episode):
        ev = Evaluator()
        summary = ev.evaluate_episode(completed_episode)
        assert isinstance(summary, EpisodeSummary)

    def test_summary_has_valid_scores(self, completed_episode):
        ev = Evaluator()
        summary = ev.evaluate_episode(completed_episode)
        assert 0.0 <= summary.composite_score <= 1.0
        assert 0.0 <= summary.ethical_alignment_score <= 1.0
        assert 0.0 <= summary.utility_score <= 1.0

    def test_summary_has_episode_metadata(self, completed_episode):
        ev = Evaluator()
        summary = ev.evaluate_episode(completed_episode)
        assert summary.episode_id == completed_episode.episode_id
        assert summary.total_turns == completed_episode.total_turns
        assert summary.completed is True

    def test_evaluate_batch(self, scenario):
        agent = BaselineAgent(seed=1)
        runner = DialogueRunner(agent=agent, max_turns=4, min_turns=2, seed=1)
        episodes = [runner.run(scenario) for _ in range(3)]
        ev = Evaluator()
        summaries = ev.evaluate_batch(episodes)
        assert len(summaries) == 3

    def test_summaries_to_dataframe(self, scenario):
        agent = BaselineAgent(seed=2)
        runner = DialogueRunner(agent=agent, max_turns=3, min_turns=1, seed=2)
        episodes = [runner.run(scenario) for _ in range(2)]
        ev = Evaluator()
        summaries = ev.evaluate_batch(episodes)
        df = ev.summaries_to_dataframe(summaries)
        assert len(df) == 2
        assert "composite_score" in df.columns

    def test_all_risk_metrics_in_range(self, completed_episode):
        ev = Evaluator()
        summary = ev.evaluate_episode(completed_episode)
        assert 0.0 <= summary.mean_coercion_risk <= 1.0
        assert 0.0 <= summary.mean_leading_question_risk <= 1.0
        assert 0.0 <= summary.mean_pressure_escalation_risk <= 1.0

    def test_all_quality_metrics_in_range(self, completed_episode):
        ev = Evaluator()
        summary = ev.evaluate_episode(completed_episode)
        assert 0.0 <= summary.mean_empathy_score <= 1.0
        assert 0.0 <= summary.mean_neutrality_score <= 1.0
        assert 0.0 <= summary.mean_information_yield <= 1.0


class TestMetricsComputer:
    """Tests for MetricsComputer."""

    def setup_method(self):
        self.mc = MetricsComputer()

    def test_compute_turn_metrics_returns_turnmetrics(self, scenario):
        from norm_dialogue_framework.agents.base_agent import DialogueState

        checker = RuleChecker()
        utterance = "Could you describe what you observed in your own words?"
        rule_result = checker.check(utterance)
        state = DialogueState(scenario=scenario, trust_level=0.6, stress_level=0.3)
        metrics = self.mc.compute_turn_metrics(utterance, rule_result, state)
        assert isinstance(metrics, TurnMetrics)
        assert 0.0 <= metrics.composite_score <= 1.0

    def test_coercive_utterance_has_high_coercion_score(self, scenario):
        from norm_dialogue_framework.agents.base_agent import DialogueState

        checker = RuleChecker()
        utterance = "You must admit what you did right now!"
        rule_result = checker.check(utterance)
        state = DialogueState(scenario=scenario, trust_level=0.5, stress_level=0.5)
        metrics = self.mc.compute_turn_metrics(utterance, rule_result, state)
        assert metrics.coercion_risk > 0.5


class TestRewardModel:
    """Tests for RewardModel."""

    def test_reward_in_range(self, scenario):
        from norm_dialogue_framework.agents.base_agent import DialogueState

        rm = RewardModel()
        state = DialogueState(scenario=scenario, trust_level=0.6, stress_level=0.3)
        score = rm.score("Could you describe what happened in your own words?", state)
        assert 0.0 <= score <= 1.0

    def test_coercive_utterance_has_lower_reward(self, scenario):
        from norm_dialogue_framework.agents.base_agent import DialogueState

        rm = RewardModel()
        state = DialogueState(scenario=scenario, trust_level=0.6, stress_level=0.3)
        safe_score = rm.score("Could you tell me what you observed?", state)
        coercive_score = rm.score("You must admit it. Don't lie. Confess now.", state)
        assert safe_score > coercive_score

    def test_rank_candidates(self, scenario):
        from norm_dialogue_framework.agents.base_agent import DialogueState

        rm = RewardModel()
        state = DialogueState(scenario=scenario, trust_level=0.6, stress_level=0.3)
        candidates = [
            "Could you describe what you saw?",
            "You must tell me everything now!",
            "In your own words, what happened?",
        ]
        ranked = rm.rank_candidates(candidates, state)
        assert len(ranked) == 3
        # Verify sorted descending
        scores = [r[1] for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_get_detailed_scores(self, scenario):
        from norm_dialogue_framework.agents.base_agent import DialogueState

        rm = RewardModel()
        state = DialogueState(scenario=scenario, trust_level=0.6, stress_level=0.3)
        details = rm.get_detailed_scores("What did you observe?", state)
        assert "composite_reward" in details
        assert "ethical_alignment_score" in details
        assert "utility_score" in details
