from abc import ABC, abstractmethod
from typing import List, Tuple

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
    def decide(self, state: MotivationalState, candidates: List[Action]) -> Tuple[Action, MotivationalState]:
        """
        The endofunctor application.
        Scores each candidate action under the updated goals and modulators[cite: 315].
        Returns the chosen action and the resulting state (G + \Delta G, M)[cite: 120, 169].
        """
        pass