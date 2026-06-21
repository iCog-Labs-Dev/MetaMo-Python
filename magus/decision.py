import numpy as np
from typing import List, Tuple

# Assuming these are available in your python path
from core.state import MotivationalState, Action
from core.config import (
    G_CURIO,
    G_ETHIC,
    G_HELP,
    G_IND,
    G_NOVEL,
    G_SELF,
    G_SOC, 
    G_TRANS,
    LAMBDA_IND,
    LAMBDA_TRANS,
    M_AROUSAL,
    M_APPROACH,
    M_RESOLUTION,
    M_SECURING,
    M_THRESHOLD,
    M_VALENCE,
    NUM_GOALS
)
from category.functors import DecisionMonad

DELTA_G_MAX_STEP = 0.1
ALIGNMENT_UPDATE_GAIN = 0.05
OVERGOAL_UPDATE_GAIN = 0.02
RISK_STABILIZATION_GAIN = 0.03


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def positive_part(value: float) -> float:
    return max(0.0, value)


def relevant_modulator(state: MotivationalState, goal_idx: int) -> float:
    """
    Maps each primary goal to the modulator most relevant to the paper's research-assistant
    specialization.
    """
    help_goal = G_HELP
    curiosity_goal = G_CURIO
    novelty_goal = G_NOVEL
    self_goal = G_SELF
    ethic_goal = G_ETHIC
    social_goal = G_SOC

    match goal_idx:
        case _ if goal_idx == help_goal:
            return state.M[M_RESOLUTION]
        case _ if goal_idx == curiosity_goal:
            return state.M[M_AROUSAL]
        case _ if goal_idx == novelty_goal:
            return state.M[M_APPROACH]
        case _ if goal_idx == self_goal:
            return (state.M[M_APPROACH] + state.M[M_RESOLUTION]) / 2.0
        case _ if goal_idx == ethic_goal:
            return (state.M[M_THRESHOLD] + state.M[M_SECURING]) / 2.0
        case _ if goal_idx == social_goal:
            return (state.M[M_VALENCE] + state.M[M_APPROACH]) / 2.0
        case _:
            return 0.0


def overgoal_support(goal_idx: int, g_ind: float, g_trans: float) -> float:
    """
    Encodes the paper's narrative coupling:
    help/ethics lean on individuation, curiosity/novelty on transcendence,
    self and social are guided by both meta-drives.
    """
    ind_support = sigmoid((g_ind - 0.5) * 6.0)
    trans_support = sigmoid((g_trans - 0.5) * 6.0)

    if goal_idx in [G_HELP, G_ETHIC]:
        return 0.5 + 0.5 * ind_support
    if goal_idx in [G_CURIO, G_NOVEL]:
        return 0.5 + 0.5 * trans_support
    if goal_idx in [G_SELF, G_SOC]:
        return 0.5 + 0.25 * (ind_support + trans_support)
    return 1.0


def normalized_growth_signal(candidate: Action) -> float:
    """
    Measures whether a candidate supports beneficial exploratory growth.
    """
    exploratory_alignment = np.mean(
        [
            positive_part(candidate.goal_correlations[G_CURIO]),
            positive_part(candidate.goal_correlations[G_NOVEL]),
            positive_part(candidate.goal_correlations[G_SELF]),
        ]
    )
    return float(exploratory_alignment)


def proposed_goal_update(state: MotivationalState, candidate: Action) -> np.ndarray:
    """
    Derive Delta G from the selected action under the current motivational state.

    The paper places the goal update inside the MAGUS decision step. Keeping it here
    prevents fallback candidates from carrying hand-authored update vectors.
    """
    delta = np.zeros(NUM_GOALS, dtype=float)
    positive_alignment = np.clip(candidate.goal_correlations, 0.0, 1.0)

    for goal_idx in range(2, NUM_GOALS):
        alignment = positive_alignment[goal_idx]
        if alignment == 0.0:
            continue
        need = 1.0 - state.G[goal_idx]
        modulator_weight = relevant_modulator(state, goal_idx)
        delta[goal_idx] = ALIGNMENT_UPDATE_GAIN * alignment * need * modulator_weight

    caution_signal = (state.M[M_THRESHOLD] + state.M[M_SECURING]) / 2.0
    delta[G_IND] = RISK_STABILIZATION_GAIN * candidate.risk_estimate * caution_signal
    delta[G_TRANS] = OVERGOAL_UPDATE_GAIN * normalized_growth_signal(candidate) * state.M[M_APPROACH]

    return np.clip(delta, -DELTA_G_MAX_STEP, DELTA_G_MAX_STEP)


class MagusDecision(DecisionMonad):
    """
    Implements the MAGUS hierarchical decision layer as the Monad (\mathbb{D}).
    Scores candidate actions using goal-specific modulators and the dual overgoals,
    then returns the proposed goal update selected by the monad.
    """

    def unit(self, state: MotivationalState) -> MotivationalState:
        """
        The monadic unit (\eta). 
        Injects the state into the monadic context without altering it.
        """
        return state

    def score_candidate(self, state: MotivationalState, candidate: Action) -> float:
        """
        Score a single candidate action under the current motivational state.
        """
        # Extract the dual overgoals that govern decision constraints
        g_ind = state.G[G_IND]      # Individuation: enforces safety and caution
        g_trans = state.G[G_TRANS]  # Transcendence: encourages growth and exploration
        caution_signal = (state.M[M_THRESHOLD] + state.M[M_SECURING]) / 2.0
        growth_signal = (state.M[M_AROUSAL] + state.M[M_APPROACH]) / 2.0

        base_score = 0.0
        conflict_penalty = 0.0

        for i in range(2, NUM_GOALS):
            goal_weight = state.G[i]
            modulator_weight = relevant_modulator(state, i)
            meta_support = overgoal_support(i, g_ind, g_trans)
            base_score += goal_weight * modulator_weight * meta_support * candidate.goal_correlations[i]

        risky_curiosity_conflict = (
            positive_part(candidate.goal_correlations[G_CURIO])
            * positive_part(-candidate.goal_correlations[G_ETHIC])
        )
        if risky_curiosity_conflict > 0.2:
            conflict_penalty = np.exp(risky_curiosity_conflict * 3.0)

        risk_penalty = LAMBDA_IND * g_ind * caution_signal * candidate.risk_estimate
        growth_reward = LAMBDA_TRANS * g_trans * growth_signal * normalized_growth_signal(candidate)

        return base_score - risk_penalty - conflict_penalty + growth_reward

    def decide(self, state: MotivationalState, candidates: List[Action]) -> Tuple[Action, np.ndarray]:
        """
        Scores each candidate action and returns the selected action together with its proposed
        goal update \Delta G. The pseudo-bimonad owns the finalized state transition after
        homeostatic damping and safe-region enforcement.
        """
        if not candidates:
            raise ValueError("Must provide at least one candidate action to the decision monad.")

        best_action = None
        best_score = -float('inf')

        for candidate in candidates:
            total_score = self.score_candidate(state, candidate)
            if total_score > best_score:
                best_score = total_score
                best_action = candidate
                 
        return best_action, proposed_goal_update(state, best_action)
