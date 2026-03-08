"""
Dialogue runner — orchestrates a full conversation episode.

The runner coordinates:
  1. Turn-by-turn generation by the dialogue agent.
  2. Respondent response simulation.
  3. Per-turn evaluation of norm compliance and utility.
  4. State tracking (trust, stress, goals addressed).
  5. Episode completion and transcript assembly.
"""

from __future__ import annotations

import random
from typing import Optional

from norm_dialogue_framework.agents.base_agent import BaseAgent, DialogueState
from norm_dialogue_framework.schemas import (
    DialogueEpisode,
    DialogueTurn,
    Scenario,
    Speaker,
)
from norm_dialogue_framework.simulation.respondent import RespondentSimulator
from norm_dialogue_framework.utils import get_logger

logger = get_logger(__name__)


class DialogueRunner:
    """Orchestrates a complete dialogue episode between an agent and respondent.

    Parameters
    ----------
    agent:
        The dialogue agent strategy to evaluate.
    max_turns:
        Maximum number of agent turns (respondent turns are paired, so the
        total turn count in the transcript will be up to 2 * max_turns).
    min_turns:
        Minimum number of agent turns before early stopping is allowed.
    seed:
        Random seed.
    evaluator:
        Optional per-turn evaluator.  When provided, each agent turn is
        scored immediately after generation.
    """

    def __init__(
        self,
        agent: BaseAgent,
        max_turns: int = 12,
        min_turns: int = 4,
        seed: int = 42,
        evaluator=None,
    ) -> None:
        self._agent = agent
        self._max_turns = max_turns
        self._min_turns = min_turns
        self._rng = random.Random(seed)
        self._evaluator = evaluator

    def run(
        self,
        scenario: Scenario,
        initial_trust: Optional[float] = None,
        initial_stress: Optional[float] = None,
    ) -> DialogueEpisode:
        """Run a complete dialogue episode for the given scenario.

        Parameters
        ----------
        scenario:
            Scenario definition.
        initial_trust, initial_stress:
            Optional overrides for respondent's starting state.

        Returns
        -------
        DialogueEpisode
            The completed episode with all turns and metadata.
        """
        episode = DialogueEpisode(
            scenario=scenario,
            agent_strategy=self._agent.strategy_name,  # type: ignore[arg-type]
        )

        respondent = RespondentSimulator(
            scenario=scenario,
            seed=self._rng.randint(0, 99999),
            initial_trust=initial_trust,
            initial_stress=initial_stress,
        )

        state = DialogueState(
            scenario=scenario,
            trust_level=respondent.trust_level,
            stress_level=respondent.stress_level,
        )

        logger.info(
            "Starting episode [strategy=%s, scenario=%s, profile=%s]",
            self._agent.strategy_name,
            scenario.scenario_type,
            scenario.respondent_profile,
        )

        for turn_idx in range(self._max_turns):
            # --- Agent turn ---
            agent_utterance = self._agent.generate_next_turn(state)

            agent_turn = DialogueTurn(
                turn_id=len(episode.turns),
                speaker=Speaker.AGENT,
                utterance=agent_utterance,
                respondent_trust_level=respondent.trust_level,
                respondent_stress_level=respondent.stress_level,
            )

            # Evaluate agent turn if evaluator is available
            if self._evaluator is not None:
                metrics = self._evaluator.evaluate_turn(agent_turn, state)
                agent_turn.turn_metrics = metrics

            episode.turns.append(agent_turn)
            state.history.append(agent_turn)

            # --- Respondent turn ---
            respondent_utterance = respondent.respond(agent_utterance)

            respondent_turn = DialogueTurn(
                turn_id=len(episode.turns),
                speaker=Speaker.RESPONDENT,
                utterance=respondent_utterance,
                respondent_trust_level=respondent.trust_level,
                respondent_stress_level=respondent.stress_level,
            )

            episode.turns.append(respondent_turn)
            state.history.append(respondent_turn)

            # Update dialogue state
            state.turn_number = turn_idx + 1
            state.trust_level = respondent.trust_level
            state.stress_level = respondent.stress_level

            # Track goal progress heuristically
            self._update_goals_addressed(state, respondent_utterance)

            # Early stopping: break if trust collapses or respondent disengages
            if self._should_stop_early(respondent, turn_idx):
                logger.info("Early stopping at turn %d (trust=%.2f).", turn_idx + 1, respondent.trust_level)
                break

        episode.total_turns = len([t for t in episode.turns if t.speaker == Speaker.AGENT])
        episode.final_trust_level = respondent.trust_level
        episode.final_stress_level = respondent.stress_level
        episode.completed = True

        logger.info(
            "Episode complete — %d agent turns, final_trust=%.2f, final_stress=%.2f",
            episode.total_turns,
            episode.final_trust_level,
            episode.final_stress_level,
        )
        return episode

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_goals_addressed(self, state: DialogueState, respondent_utterance: str) -> None:
        """Heuristically mark goals as addressed based on keywords in responses."""
        ru = respondent_utterance.lower()
        for goal in state.scenario.communication_goals:
            if goal in state.goals_addressed:
                continue
            keywords = [w for w in goal.lower().split() if len(w) > 3]
            if any(kw in ru for kw in keywords):
                state.goals_addressed.append(goal)

    def _should_stop_early(self, respondent: RespondentSimulator, turn_idx: int) -> bool:
        """Determine whether to stop the conversation early."""
        if turn_idx < self._min_turns - 1:
            return False
        # Stop if trust has collapsed
        if respondent.trust_level < 0.10:
            return True
        # Stop if respondent is very stressed and fatigued
        if respondent.stress_level > 0.92 and respondent._fatigue > 0.6:
            return True
        return False
