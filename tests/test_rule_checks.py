"""
Tests for rule-based norm checking logic.
"""

from __future__ import annotations

import pytest

from norm_dialogue_framework.evaluation.rule_checks import RuleChecker, RuleCheckResult


class TestRuleChecker:
    """Tests for the RuleChecker class."""

    def setup_method(self):
        self.checker = RuleChecker()

    # --- Coercion risk ---

    def test_coercion_detected_you_must(self):
        result = self.checker.check("You must tell me exactly what happened.")
        assert result.coercion_risk > 0.5

    def test_coercion_detected_admit(self):
        result = self.checker.check("You need to admit what you did.")
        assert result.coercion_risk > 0.5

    def test_no_coercion_in_neutral_question(self):
        result = self.checker.check("Could you describe what you observed in your own words?")
        assert result.coercion_risk == 0.0

    def test_coercion_detected_dont_lie(self):
        result = self.checker.check("Don't lie about what really happened.")
        assert result.coercion_risk > 0.7

    # --- Leading question risk ---

    def test_leading_detected_wasnt_it(self):
        result = self.checker.check("He was there, wasn't it clear?")
        assert result.leading_question_risk > 0.5

    def test_leading_detected_obviously(self):
        result = self.checker.check("Obviously you were aware of the situation.")
        assert result.leading_question_risk > 0.3

    def test_no_leading_in_open_question(self):
        result = self.checker.check("What did you observe during that period?")
        assert result.leading_question_risk == 0.0

    def test_leading_detected_surely(self):
        result = self.checker.check("Surely you noticed what was happening?")
        assert result.leading_question_risk > 0.5

    # --- Empathy score ---

    def test_empathy_detected_take_your_time(self):
        result = self.checker.check("Take your time — there's no rush.")
        assert result.empathy_score > 0.5

    def test_empathy_detected_i_understand(self):
        result = self.checker.check("I understand this may be difficult to talk about.")
        assert result.empathy_score > 0.3

    def test_no_empathy_in_blunt_question(self):
        result = self.checker.check("What did you see?")
        assert result.empathy_score == 0.0

    # --- Transparency score ---

    def test_transparency_detected(self):
        result = self.checker.check("The purpose of this question is to establish the timeline.")
        assert result.transparency_score > 0.5

    # --- Neutrality score ---

    def test_neutrality_reduced_by_judgmental_language(self):
        result = self.checker.check("That was completely unacceptable behaviour.")
        assert result.neutrality_score < 0.5

    def test_neutral_language_preserves_score(self):
        result = self.checker.check("Could you tell me more about what occurred?")
        assert result.neutrality_score == 1.0

    # --- Procedural fairness ---

    def test_procedural_fairness_detected(self):
        result = self.checker.check("You have the right to decline to answer any question.")
        assert result.procedural_fairness_score > 0.5

    # --- Pressure escalation ---

    def test_pressure_escalation_detected_across_turns(self):
        checker = RuleChecker()
        # Simulate escalating coercive turns
        checker.check("Tell me what happened.")
        checker.check("You must tell me. We already know.")
        result = checker.check("Admit what you did. Don't lie.")
        assert result.pressure_escalation_risk > 0.4

    def test_no_escalation_with_safe_turns(self):
        checker = RuleChecker()
        checker.check("Could you describe what you observed?")
        checker.check("Thank you. Can you elaborate on that?")
        result = checker.check("In your own words, what happened next?")
        assert result.pressure_escalation_risk < 0.3

    # --- RuleCheckResult structure ---

    def test_result_is_dataclass(self):
        result = self.checker.check("Could you tell me what you saw?")
        assert isinstance(result, RuleCheckResult)
        assert 0.0 <= result.coercion_risk <= 1.0
        assert 0.0 <= result.leading_question_risk <= 1.0
        assert 0.0 <= result.empathy_score <= 1.0
        assert 0.0 <= result.neutrality_score <= 1.0

    def test_reset_clears_history(self):
        checker = RuleChecker()
        checker.check("You must tell me. Admit it now.")
        checker.reset_history()
        # After reset, escalation should be fresh
        result = checker.check("Could you describe what happened?")
        assert result.pressure_escalation_risk < 0.3
