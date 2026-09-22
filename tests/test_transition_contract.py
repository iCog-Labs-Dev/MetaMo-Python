import unittest

import numpy as np

from category.bimonad import MetaMoPseudoBimonad
from category.functors import AppraisalComonad, DecisionMonad
from category.laws import LawMode, RuntimeLawPolicy
from core.schema import MagusGoalSchema, ModulatorSchema, MotivationSchema
from core.state import Action, MotivationalState


SCHEMA = MotivationSchema(
    goals=MagusGoalSchema(primary_goals=("task",)),
    modulators=ModulatorSchema(),
)


def make_state(individuation=0.8):
    return MotivationalState(
        G=np.array([individuation, 0.4, 0.6]),
        M=np.full(SCHEMA.num_modulators, 0.5),
        schema=SCHEMA,
    )


def candidates():
    result = []
    for action_id, delta in (("after_appraisal", 0.05), ("before_appraisal", -0.05)):
        delta_g = np.zeros(SCHEMA.num_goals)
        delta_g[SCHEMA.goal_index("task")] = delta
        result.append(Action(
            id=action_id,
            goal_correlations=np.zeros(SCHEMA.num_goals),
            delta_g=delta_g,
            schema=SCHEMA,
        ))
    return result


class CountingAppraisal(AppraisalComonad):
    def __init__(self):
        self.calls = 0

    def extract(self, state):
        return state

    def appraise(self, state, stimulus):
        self.calls += 1
        result = state.copy()
        result.set_modulator("approach", 0.9)
        return result


class SwitchingDecision(DecisionMonad):
    def __init__(self):
        self.calls = 0

    def unit(self, state):
        return state

    def decide(self, state, candidates, feedback=None):
        self.calls += 1
        action = candidates[0] if state.modulator("approach") > 0.5 else candidates[1]
        return action, action.delta_g.copy()


class TransitionContractTests(unittest.TestCase):
    def make_bimonad(self, lax_mode=LawMode.OBSERVE):
        appraisal = CountingAppraisal()
        decision = SwitchingDecision()
        bimonad = MetaMoPseudoBimonad(
            appraisal,
            decision,
            law_policy=RuntimeLawPolicy(
                lax_distributive=lax_mode,
                contractive=LawMode.DISABLED,
            ),
        )
        return bimonad, appraisal, decision

    def test_normal_step_is_single_pass(self):
        bimonad, appraisal, decision = self.make_bimonad()

        action, next_state = bimonad.step(
            make_state(), object(), candidates(), record_diagnostics=False
        )

        self.assertEqual(action.id, "after_appraisal")
        self.assertEqual(appraisal.calls, 1)
        self.assertEqual(decision.calls, 1)
        self.assertTrue(bimonad.stability_policy.is_in_safe_region(next_state))

    def test_audit_reuses_left_transition_and_observes_by_default(self):
        bimonad, appraisal, decision = self.make_bimonad(LawMode.OBSERVE)

        _, _, diagnostics = bimonad.step_with_diagnostics(
            make_state(), object(), candidates(), record_diagnostics=False
        )

        self.assertEqual(appraisal.calls, 2)
        self.assertEqual(decision.calls, 2)
        self.assertFalse(diagnostics.lax_holds)
        self.assertTrue(diagnostics.laws_evaluated)
        self.assertEqual(diagnostics.law_correction_delta, 0.0)

    def test_enforced_law_applies_separate_fallback_correction(self):
        bimonad, _, _ = self.make_bimonad(LawMode.ENFORCE)

        _, _, diagnostics = bimonad.step_with_diagnostics(
            make_state(), object(), candidates(), record_diagnostics=False
        )

        self.assertFalse(diagnostics.lax_holds)
        self.assertGreater(diagnostics.law_correction_delta, 0.0)
        self.assertEqual(diagnostics.projection_delta, 0.0)

    def test_final_state_is_safe_even_when_source_is_unsafe(self):
        bimonad, _, _ = self.make_bimonad()

        _, next_state = bimonad.step(
            make_state(individuation=0.1),
            object(),
            candidates(),
            record_diagnostics=False,
        )

        self.assertTrue(bimonad.stability_policy.is_in_safe_region(next_state))


if __name__ == "__main__":
    unittest.main()
