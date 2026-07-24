import numpy as np

from core.state import MotivationalState


class DefaultParallelMergePolicy:
    """
    Schema-aware default policy for Principle 3 parallel composition.
    """

    def merge(
        self,
        state_a: MotivationalState,
        state_b: MotivationalState,
        *,
        decision=None,
        coherence_correction: float = 0.05,
    ) -> MotivationalState:
        if state_a.schema != state_b.schema:
            raise ValueError("Cannot merge states with different motivation schemas")

        schema = state_a.schema
        ind_idx = schema.goal_index(schema.goals.individuation_name)

        weight_a = state_a.G[ind_idx]
        weight_b = state_b.G[ind_idx]
        total_weight = weight_a + weight_b

        if total_weight <= 1e-9:
            base_G = (state_a.G + state_b.G) / 2.0
            base_M = (state_a.M + state_b.M) / 2.0
        else:
            base_G = ((state_a.G * weight_a) + (state_b.G * weight_b)) / total_weight
            base_M = ((state_a.M * weight_a) + (state_b.M * weight_b)) / total_weight

        profile = getattr(decision, "profile", None)
        individuation_goal_names = set(getattr(profile, "individuation_goal_names", ()))
        transcendence_goal_names = set(getattr(profile, "transcendence_goal_names", ()))
        balanced_goal_names = set(getattr(profile, "balanced_goal_names", ()))

        safety_goal_names = {
            schema.goals.individuation_name,
            *individuation_goal_names,
        }
        exploratory_goal_names = {
            schema.goals.transcendence_name,
            *transcendence_goal_names,
        }

        disagreement_G = np.abs(state_a.G - state_b.G)
        disagreement_M = np.abs(state_a.M - state_b.M)

        consensus_G = base_G.copy()
        consensus_M = base_M.copy()

        safety_goal_idx = self._goal_indices_for_names(schema, safety_goal_names)
        exploratory_goal_idx = self._goal_indices_for_names(schema, exploratory_goal_names)
        balanced_goal_idx = self._goal_indices_for_names(schema, balanced_goal_names)
        anti_goal_idx = np.array(list(range(schema.goals.anti_goal_start, schema.num_goals)), dtype=int)

        if safety_goal_idx.size:
            consensus_G[safety_goal_idx] = np.maximum(state_a.G[safety_goal_idx], state_b.G[safety_goal_idx])
        if anti_goal_idx.size:
            consensus_G[anti_goal_idx] = np.maximum(state_a.G[anti_goal_idx], state_b.G[anti_goal_idx])
        if exploratory_goal_idx.size:
            consensus_G[exploratory_goal_idx] = np.minimum(state_a.G[exploratory_goal_idx], state_b.G[exploratory_goal_idx])
        if balanced_goal_idx.size:
            consensus_G[balanced_goal_idx] = (state_a.G[balanced_goal_idx] + state_b.G[balanced_goal_idx]) / 2.0

        caution_mod_idx = self._modulator_indices_for_names(schema, {"threshold", "securing"})
        exploratory_mod_idx = self._modulator_indices_for_names(schema, {"arousal", "approach"})
        shared_mod_idx = self._modulator_indices_for_names(schema, {"valence", "resolution"})

        if caution_mod_idx.size:
            consensus_M[caution_mod_idx] = np.maximum(state_a.M[caution_mod_idx], state_b.M[caution_mod_idx])
        if exploratory_mod_idx.size:
            consensus_M[exploratory_mod_idx] = np.minimum(state_a.M[exploratory_mod_idx], state_b.M[exploratory_mod_idx])
        if shared_mod_idx.size:
            consensus_M[shared_mod_idx] = (
                (state_a.M[shared_mod_idx] + state_b.M[shared_mod_idx]) / 2.0
            )

        goal_correction_scale = np.ones_like(base_G)
        if safety_goal_idx.size:
            goal_correction_scale[safety_goal_idx] = 1.5
        if anti_goal_idx.size:
            goal_correction_scale[anti_goal_idx] = 1.5
        if exploratory_goal_idx.size:
            goal_correction_scale[exploratory_goal_idx] = 1.0
        if balanced_goal_idx.size:
            goal_correction_scale[balanced_goal_idx] = 0.8

        mod_correction_scale = np.ones_like(base_M)
        if caution_mod_idx.size:
            mod_correction_scale[caution_mod_idx] = 1.5
        if exploratory_mod_idx.size:
            mod_correction_scale[exploratory_mod_idx] = 1.0
        if shared_mod_idx.size:
            mod_correction_scale[shared_mod_idx] = 0.8

        goal_correction = np.clip(coherence_correction * disagreement_G * goal_correction_scale, 0.0, 1.0)
        mod_correction = np.clip(coherence_correction * disagreement_M * mod_correction_scale, 0.0, 1.0)

        merged_G = base_G + goal_correction * (consensus_G - base_G)
        merged_M = base_M + mod_correction * (consensus_M - base_M)

        return MotivationalState(G=merged_G, M=merged_M, schema=state_a.schema)

    @staticmethod
    def _goal_indices_for_names(schema, names: set[str]) -> np.ndarray:
        indices = [
            schema.goal_index(name)
            for name in names
            if name in schema.goal_names
        ]
        return np.array(sorted(set(indices)), dtype=int)

    @staticmethod
    def _modulator_indices_for_names(schema, names: set[str]) -> np.ndarray:
        indices = [
            schema.modulator_index(name)
            for name in names
            if name in schema.modulator_names
        ]
        return np.array(sorted(set(indices)), dtype=int)
