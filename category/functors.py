from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple
import numpy as np

from category.simulation import ReciprocalSimulationResult
from core.schema import MotivationSchema
from core.state import MotivationalState, Action

TransitionFunction = Callable[
    [MotivationalState, Any, List[Action]],
    Tuple[Action, MotivationalState],
]
StateTransform = Callable[[MotivationalState], MotivationalState]
ApplicationStimulusTransform = Callable[[Any], Any]
CandidateTransform = Callable[[List[Action]], List[Action]]


def _fit_translation_matrix(
    source_vectors: List[np.ndarray],
    target_vectors: List[np.ndarray],
    regularization: float,
) -> np.ndarray:
    
    if len(source_vectors) != len(target_vectors):
        raise ValueError("source and target vector lists must have the same length")
    if not source_vectors:
        raise ValueError("must provide at least one paired example")

    source = np.vstack(source_vectors).astype(float)
    target = np.vstack(target_vectors).astype(float)

    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source and target examples must be 2D after stacking")
    if regularization < 0.0:
        raise ValueError("regularization must be non-negative")

    lhs = source.T @ source
    if regularization > 0.0:
        lhs = lhs + np.eye(lhs.shape[0]) * regularization
    rhs = source.T @ target

    try:
        fitted = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        fitted = np.linalg.lstsq(source, target, rcond=None)[0]

    return fitted.T


class AppraisalComonad(ABC):
    """
    Abstract base class for the Appraisal Comonad (ψ).
    In MetaMo, the comonad handles application-stimulus appraisal, updating affect and modulators.
    It maps the state and an application stimulus to a new state.
    """

    @abstractmethod
    def extract(self, state: MotivationalState) -> MotivationalState:
        """
        The comonadic counit (epsilon).
        Extracts the current observable state from the comonadic context.
        """
        pass

    @abstractmethod
    def appraise(self, state: MotivationalState, stimulus: Any) -> MotivationalState:
        """
        The endofunctor application.
        Updates the modulators M based on application stimulus without altering the high-level goals G.
        Yields ψ((G, M), s) = (G, M').
        """
        pass


class DecisionMonad(ABC):
    """
    Abstract base class for the Decision Monad (D).
    In MetaMo, the monad handles goal selection and action scoring.
    It maps the state to a new goal configuration: D(X).
    """

    @abstractmethod
    def unit(self, state: MotivationalState) -> MotivationalState:
        """
        The monadic unit (eta).
        Injects a pure motivational state into the monadic decision context.
        """
        pass

    @abstractmethod
    def decide(
        self,
        state: MotivationalState,
        candidates: List[Action],
        feedback: Any = None,
    ) -> Tuple[Action, np.ndarray]:
        """
        The endofunctor application.
        Scores each candidate action under the updated goals and modulators.
        Returns the chosen action and the proposed goal update \Delta G.
        The composite operator F = D o ψ is responsible for turning this proposal into
        the finalized next motivational state.
        """
        pass

class TranslationFunctor:
    """
    Implements Principle 2: Reciprocal Motivational State Simulation.
    Maps Agent A's state into Agent B's state space for seamless hand-off.
    """
    def __init__(
        self,
        goal_translation: np.ndarray,
        modulator_translation: np.ndarray,
        target_schema: MotivationSchema | None = None,
    ):
        """
        Separate linear maps for translating goal-space and modulator-space coordinates.
        """
        goal_translation = np.asarray(goal_translation, dtype=float)
        modulator_translation = np.asarray(modulator_translation, dtype=float)

        if goal_translation.ndim != 2:
            raise ValueError("goal_translation must be a 2D matrix")
        if modulator_translation.ndim != 2:
            raise ValueError("modulator_translation must be a 2D matrix")

        self.goal_translation = goal_translation
        self.modulator_translation = modulator_translation
        self.target_schema = target_schema

    @classmethod
    def fit_from_state_pairs(
        cls,
        source_states: List[MotivationalState],
        target_states: List[MotivationalState],
        regularization: float = 1e-8,
    ) -> "TranslationFunctor":
        """
        Fit linear state-translation maps from paired hand-off examples.
        """
        if len(source_states) != len(target_states):
            raise ValueError("source_states and target_states must have the same length")
        if not source_states:
            raise ValueError("must provide at least one paired state")

        goal_translation = _fit_translation_matrix(
            [state.G for state in source_states],
            [state.G for state in target_states],
            regularization=regularization,
        )
        modulator_translation = _fit_translation_matrix(
            [state.M for state in source_states],
            [state.M for state in target_states],
            regularization=regularization,
        )

        target_schema = target_states[0].schema
        if any(state.schema != target_schema for state in target_states):
            raise ValueError("target states must share one motivation schema")

        return cls(
            goal_translation=goal_translation,
            modulator_translation=modulator_translation,
            target_schema=target_schema,
        )
        
    def simulate_peer(self, state_a: MotivationalState) -> MotivationalState:
        """
        Applies functor T to shadow another agent's motivational frame.
        """
        if self.goal_translation.shape[1] != state_a.G.shape[0]:
            raise ValueError("goal translation input dimensions do not match the state goal vector")
        if self.modulator_translation.shape[1] != state_a.M.shape[0]:
            raise ValueError("modulator translation input dimensions do not match the state modulator vector")

        target_schema = self.target_schema or state_a.schema
        if self.goal_translation.shape[0] != target_schema.num_goals:
            raise ValueError("goal translation output dimensions do not match the target schema")
        if self.modulator_translation.shape[0] != target_schema.num_modulators:
            raise ValueError("modulator translation output dimensions do not match the target schema")

        simulated_G = np.dot(self.goal_translation, state_a.G)
        simulated_M = np.dot(self.modulator_translation, state_a.M)
        
        return MotivationalState(
            G=np.clip(simulated_G, 0.0, 1.0),
            M=np.clip(simulated_M, 0.0, 1.0),
            schema=target_schema,
        )

    def reciprocal_round_trip_error(
        self,
        state_a: MotivationalState,
        inverse_translation: "TranslationFunctor",
    ) -> float:
        """
        Measures how much state is lost by translating A -> B -> A.
        A low value supports reciprocal, not merely one-way, simulation.
        """
        translated = self.simulate_peer(state_a)
        reconstructed = inverse_translation.simulate_peer(translated)
        return state_a.distance_to(reconstructed)

    def check_reciprocal_simulation(
        self,
        source_update: TransitionFunction,
        target_update: TransitionFunction,
        source_state: MotivationalState,
        stimulus: Any,
        candidates: List[Action],
        tolerance: float = 0.05,
        natural_transform: Optional[StateTransform] = None,
        stimulus_translation: Optional[ApplicationStimulusTransform] = None,
        candidate_translation: Optional[CandidateTransform] = None,
    ) -> ReciprocalSimulationResult:
        """
        Validates Principle 2 by checking the commuting update square.

        The left path computes T(F_A(x)): update the source agent, then translate.
        The right path computes F_B(T(x)): translate first, then update the target.
        """
        source_action, source_next = source_update(source_state, stimulus, candidates)

        translated_after_source_update = self.simulate_peer(source_next)
        if natural_transform is not None:
            translated_after_source_update = natural_transform(translated_after_source_update)

        translated_source_state = self.simulate_peer(source_state)
        target_stimulus = stimulus_translation(stimulus) if stimulus_translation else stimulus
        target_candidates = candidate_translation(list(candidates)) if candidate_translation else candidates
        target_action, target_after_translation_update = target_update(
            translated_source_state,
            target_stimulus,
            target_candidates,
        )

        error = translated_after_source_update.distance_to(target_after_translation_update)
        return ReciprocalSimulationResult(
            source_action=source_action,
            target_action=target_action,
            translated_after_source_update=translated_after_source_update,
            target_after_translation_update=target_after_translation_update,
            error=error,
            tolerance=tolerance,
            holds=error <= tolerance,
        )


@dataclass(frozen=True)
class AgentFrameAdapter:
    """
    Heterogeneous-agent adapter for state, stimulus, action, and output frames.
    """

    state_translation: TranslationFunctor
    stimulus_translation: Optional[ApplicationStimulusTransform] = None
    candidate_translation: Optional[CandidateTransform] = None
    natural_transform: Optional[StateTransform] = None

    def translate_state(self, state: MotivationalState) -> MotivationalState:
        return self.state_translation.simulate_peer(state)

    def translate_stimulus(self, stimulus: Any) -> Any:
        if self.stimulus_translation is None:
            return stimulus
        return self.stimulus_translation(stimulus)

    def translate_candidates(self, candidates: List[Action]) -> List[Action]:
        if self.candidate_translation is None:
            return candidates
        return self.candidate_translation(candidates)

    def check_reciprocal_simulation(
        self,
        source_update: TransitionFunction,
        target_update: TransitionFunction,
        source_state: MotivationalState,
        stimulus: Any,
        candidates: List[Action],
        tolerance: float = 0.05,
    ) -> ReciprocalSimulationResult:
        return self.state_translation.check_reciprocal_simulation(
            source_update=source_update,
            target_update=target_update,
            source_state=source_state,
            stimulus=stimulus,
            candidates=candidates,
            tolerance=tolerance,
            natural_transform=self.natural_transform,
            stimulus_translation=self.stimulus_translation,
            candidate_translation=self.candidate_translation,
        )
