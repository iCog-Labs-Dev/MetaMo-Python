from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

from core.state import MotivationalState, Stimulus, Action

class AppraisalComonad(ABC):
    """
    Abstract base class for the Appraisal Comonad (\Psi).
    In MetaMo, the comonad handles stimulus appraisal, updating affect and modulators[cite: 28, 307].
    It maps the state and a stimulus to a new state: \Psi(X \times S) -> X[cite: 314].
    """

    @abstractmethod
    def extract(self, state: MotivationalState) -> MotivationalState:
        """
        The comonadic counit (\epsilon).
        Extracts the current observable state from the comonadic context.
        """
        pass

    @abstractmethod
    def appraise(self, state: MotivationalState, stimulus: Stimulus) -> MotivationalState:
        """
        The endofunctor application.
        Updates the modulators M based on the stimulus without altering the high-level goals G[cite: 54].
        Yields \Psi((G, M), s) = (G, M')[cite: 55, 160].
        """
        pass


class DecisionMonad(ABC):
    """
    Abstract base class for the Decision Monad (\mathbb{D}).
    In MetaMo, the monad handles goal selection and action scoring[cite: 28, 307].
    It maps the state to a new goal configuration: \mathbb{D}(X).
    """

    @abstractmethod
    def unit(self, state: MotivationalState) -> MotivationalState:
        """
        The monadic unit (\eta).
        Injects a pure motivational state into the monadic decision context.
        """
        pass

    @abstractmethod
    def decide(self, state: MotivationalState, candidates: List[Action]) -> Tuple[Action, np.ndarray]:
        """
        The endofunctor application.
        Scores each candidate action under the updated goals and modulators[cite: 315].
        Returns the chosen action and the proposed goal update \Delta G[cite: 120, 169].
        The composite operator F = D \circ \Psi is responsible for turning this proposal into
        the finalized next motivational state.
        """
        pass

    # Add to category/functors.py
class TranslationFunctor:
    """
    Implements Principle 2: Reciprocal Motivational State Simulation.
    Maps Agent A's state into Agent B's state space for seamless hand-off.
    """
    def __init__(self, goal_translation: np.ndarray, modulator_translation: np.ndarray):
        """
        Separate linear maps for translating goal-space and modulator-space coordinates.
        """
        if goal_translation.ndim != 2:
            raise ValueError("goal_translation must be a 2D matrix")
        if modulator_translation.ndim != 2:
            raise ValueError("modulator_translation must be a 2D matrix")

        goal_rows, goal_cols = goal_translation.shape
        mod_rows, mod_cols = modulator_translation.shape

        if goal_rows != goal_cols:
            raise ValueError("goal_translation must be square for same-space peer simulation")
        if mod_rows != mod_cols:
            raise ValueError("modulator_translation must be square for same-space peer simulation")

        self.goal_translation = goal_translation
        self.modulator_translation = modulator_translation
        
    def simulate_peer(self, state_a: MotivationalState) -> MotivationalState:
        """
        Applies functor T to shadow another agent's motivational frame.
        """
        if self.goal_translation.shape[1] != state_a.G.shape[0]:
            raise ValueError("goal translation dimensions do not match the state goal vector")
        if self.modulator_translation.shape[1] != state_a.M.shape[0]:
            raise ValueError("modulator translation dimensions do not match the state modulator vector")

        simulated_G = np.dot(self.goal_translation, state_a.G)
        simulated_M = np.dot(self.modulator_translation, state_a.M)
        
        return MotivationalState(
            G=np.clip(simulated_G, 0.0, 1.0),
            M=np.clip(simulated_M, 0.0, 1.0)
        )