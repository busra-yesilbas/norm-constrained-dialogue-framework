"""
Tests for the dialogue runner and respondent simulator.
"""

from __future__ import annotations

import pytest

from norm_dialogue_framework.agents.baseline_agent import BaselineAgent
from norm_dialogue_framework.agents.candidate_selector_agent import CandidateSelectorAgent
from norm_dialogue_framework.agents.rule_augmented_agent import RuleAugmentedAgent
from norm_dialogue_framework.data.synthetic_generator import SyntheticScenarioGenerator
from norm_dialogue_framework.evaluation.evaluator import Evaluator
from norm_dialogue_framework.schemas import DialogueEpisode, Speaker
from norm_dialogue_framework.simulation.dialogue_runner import DialogueRunner
from norm_dialogue_framework.simulation.respondent import RespondentSimulator


@pytest.fixture
def scenario():
    gen = SyntheticScenarioGenerator(seed=42)
    return gen.generate(n=1)[0]


@pytest.fixture
def baseline_runner():
    agent = BaselineAgent(seed=42)
    return DialogueRunner(agent=agent, max_turns=5, min_turns=2, seed=42)


class TestDialogueRunner:
    """Tests for DialogueRunner."""

    def test_run_returns_episode(self, baseline_runner, scenario):
        episode = baseline_runner.run(scenario)
        assert isinstance(episode, DialogueEpisode)

    def test_episode_has_turns(self, baseline_runner, scenario):
        episode = baseline_runner.run(scenario)
        assert len(episode.turns) > 0

    def test_episode_alternates_speakers(self, baseline_runner, scenario):
        episode = baseline_runner.run(scenario)
        speakers = [t.speaker for t in episode.turns]
        for i in range(len(speakers) - 1):
            assert speakers[i] != speakers[i + 1], (
                f"Consecutive turns from same speaker at index {i}"
            )

    def test_episode_starts_with_agent(self, baseline_runner, scenario):
        episode = baseline_runner.run(scenario)
        assert episode.turns[0].speaker == Speaker.AGENT

    def test_episode_max_turns_respected(self, scenario):
        agent = BaselineAgent(seed=42)
        runner = DialogueRunner(agent=agent, max_turns=3, min_turns=1, seed=42)
        episode = runner.run(scenario)
        agent_turns = [t for t in episode.turns if t.speaker == Speaker.AGENT]
        assert len(agent_turns) <= 3

    def test_episode_completed_flag(self, baseline_runner, scenario):
        episode = baseline_runner.run(scenario)
        assert episode.completed is True

    def test_final_trust_and_stress_are_set(self, baseline_runner, scenario):
        episode = baseline_runner.run(scenario)
        assert episode.final_trust_level is not None
        assert 0.0 <= episode.final_trust_level <= 1.0
        assert episode.final_stress_level is not None
        assert 0.0 <= episode.final_stress_level <= 1.0

    def test_all_utterances_are_strings(self, baseline_runner, scenario):
        episode = baseline_runner.run(scenario)
        for turn in episode.turns:
            assert isinstance(turn.utterance, str)
            assert len(turn.utterance) > 0

    def test_run_with_evaluator(self, scenario):
        agent = BaselineAgent(seed=42)
        evaluator = Evaluator()
        runner = DialogueRunner(
            agent=agent, max_turns=4, min_turns=2, seed=42, evaluator=evaluator
        )
        episode = runner.run(scenario)
        agent_turns = [t for t in episode.turns if t.speaker == Speaker.AGENT]
        for turn in agent_turns:
            assert turn.turn_metrics is not None

    def test_candidate_selector_runs(self, scenario):
        agent = CandidateSelectorAgent(n_candidates=3, seed=42)
        runner = DialogueRunner(agent=agent, max_turns=4, min_turns=2, seed=42)
        episode = runner.run(scenario)
        assert episode.completed is True
        assert len(episode.turns) > 0

    def test_rule_augmented_runs(self, scenario):
        agent = RuleAugmentedAgent(seed=42)
        runner = DialogueRunner(agent=agent, max_turns=4, min_turns=2, seed=42)
        episode = runner.run(scenario)
        assert episode.completed is True

    def test_initial_trust_override(self, scenario):
        agent = BaselineAgent(seed=42)
        runner = DialogueRunner(agent=agent, max_turns=3, min_turns=1, seed=42)
        episode = runner.run(scenario, initial_trust=0.9, initial_stress=0.1)
        assert episode.completed is True


class TestRespondentSimulator:
    """Tests for RespondentSimulator."""

    def test_responds_with_string(self, scenario):
        sim = RespondentSimulator(scenario=scenario, seed=42)
        response = sim.respond("Could you tell me what happened?")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_trust_level_in_range(self, scenario):
        sim = RespondentSimulator(scenario=scenario, seed=42)
        assert 0.0 <= sim.trust_level <= 1.0

    def test_stress_level_in_range(self, scenario):
        sim = RespondentSimulator(scenario=scenario, seed=42)
        assert 0.0 <= sim.stress_level <= 1.0

    def test_coercive_utterance_reduces_trust(self, scenario):
        # Use a resistant or trauma_sensitive profile for clearer effect
        from norm_dialogue_framework.data.synthetic_generator import SyntheticScenarioGenerator
        gen = SyntheticScenarioGenerator(
            seed=42, respondent_profiles=["trauma_sensitive"]
        )
        sc = gen.generate(n=1)[0]
        sim = RespondentSimulator(scenario=sc, seed=42, initial_trust=0.7, initial_stress=0.4)
        initial_trust = sim.trust_level
        sim.respond("You must admit what really happened. Don't lie.")
        assert sim.trust_level <= initial_trust

    def test_empathetic_utterance_can_improve_trust(self, scenario):
        sim = RespondentSimulator(scenario=scenario, seed=42, initial_trust=0.4, initial_stress=0.6)
        initial_trust = sim.trust_level
        for _ in range(3):
            sim.respond("I understand this is difficult. Take your time — in your own words.")
        assert sim.trust_level >= initial_trust - 0.05  # should not degrade significantly

    def test_multiple_turns(self, scenario):
        sim = RespondentSimulator(scenario=scenario, seed=42)
        for i in range(5):
            resp = sim.respond(f"Turn {i}: Could you describe what you observed?")
            assert isinstance(resp, str)
