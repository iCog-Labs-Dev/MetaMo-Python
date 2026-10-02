"""
Logs and computes all evaluation metrics.
"""

import numpy as np
from dataclasses import dataclass, field
from collections import Counter


RECOVERY_CAP = 50
RECOVERY_L   = 3          


@dataclass
class EpisodeLog:
    minerals_collected: int   = 0
    minerals_spawned:   int   = 0        
    total_steps:        int   = 0
    total_reward:       float = 0.0
    lava_steps:         int   = 0      
    movement_steps:     int   = 0

    mot_srv_flags:      list  = field(default_factory=list)   # MetaMo: not in_safe_region(mot)
    env_srv_flags:      list  = field(default_factory=list)   # Baseline: in lava-band proxy
    mot_boundary_flags: list  = field(default_factory=list)   # MetaMo: inside boundary band B_eta
    mot_pressure_log:   list  = field(default_factory=list)   # MetaMo: boundary pressure [0, 1]
    appraisal_influence_flags: list = field(default_factory=list)
    appraisal_score_shift_log: list = field(default_factory=list)
    exploration_flags: list = field(default_factory=list)
    risk_penalty_log: list = field(default_factory=list)
    selector_influence_flags: list = field(default_factory=list)
    hard_safety_intervention_flags: list = field(default_factory=list)
    q_regret_log: list = field(default_factory=list)
    task_score_log: list = field(default_factory=list)
    safety_score_log: list = field(default_factory=list)
    task_safety_disagreement_log: list = field(default_factory=list)
    q_shortlist_size_log: list = field(default_factory=list)
    q_proposed_lava_flags: list = field(default_factory=list)
    executed_lava_entry_flags: list = field(default_factory=list)
    q_proposal_blocked_flags: list = field(default_factory=list)
    executed_changed_q_flags: list = field(default_factory=list)
    q_proposed_risk_log: list = field(default_factory=list)
    executed_risk_log: list = field(default_factory=list)
    q_proposed_visit_count_log: list = field(default_factory=list)
    q_proposed_value_log: list = field(default_factory=list)
    compositionality_error_log: list = field(default_factory=list)
    compositionality_holds_flags: list = field(default_factory=list)
    compositionality_goal_error_log: list = field(default_factory=list)
    compositionality_modulator_error_log: list = field(default_factory=list)
    compositionality_action_holds_flags: list = field(default_factory=list)
    compositionality_goal_coordinate_log: list = field(default_factory=list)
    compositionality_modulator_coordinate_log: list = field(default_factory=list)

    unsafe_flags:       list  = field(default_factory=list)   # both: in_lava OR lava_dist ≤ band
    arousal_log:        list  = field(default_factory=list)   # MetaMo only
    safety_log:         list  = field(default_factory=list)   # MetaMo only
    individuation_log:  list  = field(default_factory=list)   # MetaMo only
    transcendence_log:  list  = field(default_factory=list)   # MetaMo only
    energy_log:         list  = field(default_factory=list)
    survived:           bool  = True

    # Computed metrics
 
    def completion_rate(self) -> float:
        """Return the fraction of spawned minerals successfully collected."""
        if self.minerals_spawned == 0:
            return 0.0
        return self.minerals_collected / self.minerals_spawned

    def mot_srv_rate(self) -> float:
        """MetaMo motivational SRV rate — internal signal only."""
        if not self.mot_srv_flags:
            return 0.0
        return sum(self.mot_srv_flags) / len(self.mot_srv_flags)

    def env_srv_rate(self) -> float:
        """Baseline environmental SRV proxy — danger-band exposure."""
        if not self.env_srv_flags:
            return 0.0
        return sum(self.env_srv_flags) / len(self.env_srv_flags)

    def srv_rate(self) -> float:
        """
        Unified accessor used by MetricsCollector.summary().
        Returns the appropriate SRV rate depending on which flags were populated.
        MetaMo episodes populate mot_srv_flags; baseline episodes populate env_srv_flags.
        """
        if self.mot_srv_flags:
            return self.mot_srv_rate()
        return self.env_srv_rate()

    def unsafe_rate(self) -> float:
        """Fraction of steps in lava or the danger band. Same definition for both agents."""
        if not self.unsafe_flags:
            return 0.0
        return sum(self.unsafe_flags) / len(self.unsafe_flags)

    def mot_boundary_rate(self) -> float:
        """Fraction of MetaMo steps in the motivational boundary band."""
        if not self.mot_boundary_flags:
            return 0.0
        return sum(self.mot_boundary_flags) / len(self.mot_boundary_flags)

    def mean_mot_pressure(self) -> float:
        """Mean MetaMo boundary pressure, where 0 is comfortable and 1 is outside/on edge."""
        if not self.mot_pressure_log:
            return 0.0
        return float(np.mean(self.mot_pressure_log))

    def appraisal_influence_rate(self) -> float:
        """Fraction of greedy decisions changed by rule versus neutral appraisal."""
        if not self.appraisal_influence_flags:
            return 0.0
        return float(np.mean(self.appraisal_influence_flags))

    def mean_appraisal_score_shift(self) -> float:
        """Mean maximum candidate-score change attributable to appraisal."""
        if not self.appraisal_score_shift_log:
            return 0.0
        return float(np.mean(self.appraisal_score_shift_log))

    def exploration_rate(self) -> float:
        if not self.exploration_flags:
            return 0.0
        return float(np.mean(self.exploration_flags))

    def mean_risk_penalty(self) -> float:
        if not self.risk_penalty_log:
            return 0.0
        return float(np.mean(self.risk_penalty_log))

    def selector_influence_rate(self) -> float:
        """Fraction of greedy decisions changed relative to Q alone."""
        return (
            float(np.mean(self.selector_influence_flags))
            if self.selector_influence_flags
            else 0.0
        )

    def hard_safety_intervention_rate(self) -> float:
        return (
            float(np.mean(self.hard_safety_intervention_flags))
            if self.hard_safety_intervention_flags
            else 0.0
        )

    def mean_q_regret(self) -> float:
        return float(np.mean(self.q_regret_log)) if self.q_regret_log else 0.0

    def mean_task_score(self) -> float:
        return float(np.mean(self.task_score_log)) if self.task_score_log else 0.0

    def mean_safety_score(self) -> float:
        return float(np.mean(self.safety_score_log)) if self.safety_score_log else 0.0

    def mean_task_safety_disagreement(self) -> float:
        return (
            float(np.mean(self.task_safety_disagreement_log))
            if self.task_safety_disagreement_log
            else 0.0
        )

    def mean_q_shortlist_size(self) -> float:
        return (
            float(np.mean(self.q_shortlist_size_log))
            if self.q_shortlist_size_log
            else 0.0
        )

    def q_proposed_lava_rate(self) -> float:
        return (
            float(np.mean(self.q_proposed_lava_flags))
            if self.q_proposed_lava_flags
            else 0.0
        )

    def executed_lava_entry_rate(self) -> float:
        return (
            float(np.mean(self.executed_lava_entry_flags))
            if self.executed_lava_entry_flags
            else 0.0
        )

    def q_proposal_blocked_rate(self) -> float:
        return (
            float(np.mean(self.q_proposal_blocked_flags))
            if self.q_proposal_blocked_flags
            else 0.0
        )

    def executed_changed_q_rate(self) -> float:
        return (
            float(np.mean(self.executed_changed_q_flags))
            if self.executed_changed_q_flags
            else 0.0
        )

    def mean_q_proposed_risk(self) -> float:
        return (
            float(np.mean(self.q_proposed_risk_log))
            if self.q_proposed_risk_log
            else 0.0
        )

    def mean_executed_risk(self) -> float:
        return (
            float(np.mean(self.executed_risk_log))
            if self.executed_risk_log
            else 0.0
        )

    def mean_q_proposed_visit_count(self) -> float:
        return (
            float(np.mean(self.q_proposed_visit_count_log))
            if self.q_proposed_visit_count_log
            else 0.0
        )

    def mean_q_proposed_value(self) -> float:
        return (
            float(np.mean(self.q_proposed_value_log))
            if self.q_proposed_value_log
            else 0.0
        )

    def mean_compositionality_error(self) -> float:
        return (
            float(np.mean(self.compositionality_error_log))
            if self.compositionality_error_log
            else 0.0
        )

    def median_compositionality_error(self) -> float:
        return (
            float(np.median(self.compositionality_error_log))
            if self.compositionality_error_log
            else 0.0
        )

    def p95_compositionality_error(self) -> float:
        return (
            float(np.percentile(self.compositionality_error_log, 95))
            if self.compositionality_error_log
            else 0.0
        )

    def max_compositionality_error(self) -> float:
        return (
            float(np.max(self.compositionality_error_log))
            if self.compositionality_error_log
            else 0.0
        )

    def mean_compositionality_goal_error(self) -> float:
        return (
            float(np.mean(self.compositionality_goal_error_log))
            if self.compositionality_goal_error_log
            else 0.0
        )

    def mean_compositionality_modulator_error(self) -> float:
        return (
            float(np.mean(self.compositionality_modulator_error_log))
            if self.compositionality_modulator_error_log
            else 0.0
        )

    def compositionality_action_holds_rate(self) -> float:
        return (
            float(np.mean(self.compositionality_action_holds_flags))
            if self.compositionality_action_holds_flags
            else 0.0
        )

    @staticmethod
    def _dominant_coordinate(names: list) -> str:
        return Counter(names).most_common(1)[0][0] if names else ""

    def dominant_compositionality_goal_coordinate(self) -> str:
        return self._dominant_coordinate(self.compositionality_goal_coordinate_log)

    def dominant_compositionality_modulator_coordinate(self) -> str:
        return self._dominant_coordinate(self.compositionality_modulator_coordinate_log)

    def compositionality_holds_rate(self) -> float:
        return (
            float(np.mean(self.compositionality_holds_flags))
            if self.compositionality_holds_flags
            else 0.0
        )

    def recovery_time(self) -> float:
        """
        Environmental recovery time from unsafe-zone exposure.

        RT(t0) = min{tau >= 0 : RECOVERY_L consecutive safe steps after unsafe-zone
        exposure at t0}

        Uses unsafe_flags for both agents, so the printed recovery metric is
        comparable between Baseline and MetaMo.
        """
        return self._recovery_time_from_flags(self.unsafe_flags)

    def mot_recovery_time(self) -> float:
        """
        Motivational recovery time from actual safe-region violations.

        With projection enabled, this should usually be 0 because actual
        safe-region violations are prevented by design.
        """
        return self._recovery_time_from_flags(self.mot_srv_flags)

    def mot_boundary_recovery_time(self) -> float:
        """Motivational recovery time from boundary-band pressure."""
        return self._recovery_time_from_flags(self.mot_boundary_flags)

    def _recovery_time_from_flags(self, flags: list) -> float:
        """
        Returns average RT over all violation bouts, capped at RECOVERY_CAP.
        If no violations, returns 0.0.
        If violations exist but none recovered, returns RECOVERY_CAP.
        """
        if not flags:
            return 0.0

        rts = []
        i = 0
        while i < len(flags):
            if flags[i]:                          
                t0 = i
                j  = i + 1
                consec_safe = 0
                recovered   = False
                while j < len(flags):
                    if not flags[j]:
                        consec_safe += 1
                        if consec_safe >= RECOVERY_L:
                            rts.append(min(j - t0, RECOVERY_CAP))
                            recovered = True
                            break
                    else:
                        consec_safe = 0
                    j += 1
                if not recovered:
                    rts.append(RECOVERY_CAP)
                i = j + 1
            else:
                i += 1

        return float(np.mean(rts)) if rts else 0.0

    def lava_rate(self) -> float:
        """Return the fraction of episode steps spent inside lava."""
        if self.total_steps == 0:
            return 0.0
        return self.lava_steps / self.total_steps

    def final_energy(self) -> float:
        """Return the last recorded energy for this episode."""
        if not self.energy_log:
            return 0.0
        return float(self.energy_log[-1])

    def survival_rate(self) -> float:
        """Return 1.0 when the episode ended with the agent alive."""
        return 1.0 if self.survived else 0.0

    def path_efficiency(self) -> float:
        """Minerals collected per movement step."""
        steps = self.movement_steps or self.total_steps
        if steps == 0:
            return 0.0
        return self.minerals_collected / steps


class MetricsCollector:
    """Accumulates EpisodeLogs across episodes and computes summary stats."""

    def __init__(self, label: str):
        self.label    = label
        self.episodes: list[EpisodeLog] = []

    def add(self, ep: EpisodeLog):
        self.episodes.append(ep)

    @staticmethod
    def _stat(values: list[float]) -> dict:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}

        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if arr.size <= 1:
            margin = 0.0
        else:
            margin = float(1.96 * (std / np.sqrt(arr.size)))
        return {
            "mean": mean,
            "std": std,
            "ci95_low": mean - margin,
            "ci95_high": mean + margin,
        }

    def summary(self) -> dict:
        """
       Compute aggregate statistics across all recorded evaluation episodes.
       Returns means and standard deviations for each evaluation metric.
       """
        n = len(self.episodes)
        if n == 0:
            return {}

        cr     = [e.completion_rate() for e in self.episodes]
        lr     = [e.lava_rate()       for e in self.episodes]
        tr     = [e.total_reward      for e in self.episodes]
        srv    = [e.srv_rate()        for e in self.episodes]
        unsafe = [e.unsafe_rate()     for e in self.episodes]
        rt     = [e.recovery_time()   for e in self.episodes]
        final_energy = [e.final_energy() for e in self.episodes]
        survival = [e.survival_rate() for e in self.episodes]
        path_efficiency = [e.path_efficiency() for e in self.episodes]
        minerals_collected = [float(e.minerals_collected) for e in self.episodes]

        mot_srv = [e.mot_srv_rate() for e in self.episodes if e.mot_srv_flags]
        env_srv = [e.env_srv_rate() for e in self.episodes if e.env_srv_flags]
        mot_boundary = [e.mot_boundary_rate() for e in self.episodes if e.mot_boundary_flags]
        mot_pressure = [e.mean_mot_pressure() for e in self.episodes if e.mot_pressure_log]
        mot_boundary_rt = [
            e.mot_boundary_recovery_time()
            for e in self.episodes
            if e.mot_boundary_flags
        ]
        appraisal_influence = [
            e.appraisal_influence_rate()
            for e in self.episodes
            if e.appraisal_influence_flags
        ]
        appraisal_score_shift = [
            e.mean_appraisal_score_shift()
            for e in self.episodes
            if e.appraisal_score_shift_log
        ]
        exploration = [
            e.exploration_rate() for e in self.episodes if e.exploration_flags
        ]
        risk_penalty = [
            e.mean_risk_penalty() for e in self.episodes if e.risk_penalty_log
        ]
        selector_influence = [
            e.selector_influence_rate()
            for e in self.episodes
            if e.selector_influence_flags
        ]
        hard_safety_intervention = [
            e.hard_safety_intervention_rate()
            for e in self.episodes
            if e.hard_safety_intervention_flags
        ]
        q_regret = [e.mean_q_regret() for e in self.episodes if e.q_regret_log]
        task_score = [e.mean_task_score() for e in self.episodes if e.task_score_log]
        safety_score = [
            e.mean_safety_score() for e in self.episodes if e.safety_score_log
        ]
        task_safety_disagreement = [
            e.mean_task_safety_disagreement()
            for e in self.episodes
            if e.task_safety_disagreement_log
        ]
        q_shortlist_size = [
            e.mean_q_shortlist_size()
            for e in self.episodes
            if e.q_shortlist_size_log
        ]
        q_proposed_lava = [
            e.q_proposed_lava_rate()
            for e in self.episodes
            if e.q_proposed_lava_flags
        ]
        executed_lava_entry = [
            e.executed_lava_entry_rate()
            for e in self.episodes
            if e.executed_lava_entry_flags
        ]
        q_proposal_blocked = [
            e.q_proposal_blocked_rate()
            for e in self.episodes
            if e.q_proposal_blocked_flags
        ]
        executed_changed_q = [
            e.executed_changed_q_rate()
            for e in self.episodes
            if e.executed_changed_q_flags
        ]
        q_proposed_risk = [
            e.mean_q_proposed_risk()
            for e in self.episodes
            if e.q_proposed_risk_log
        ]
        executed_risk = [
            e.mean_executed_risk()
            for e in self.episodes
            if e.executed_risk_log
        ]
        q_proposed_visit_count = [
            e.mean_q_proposed_visit_count()
            for e in self.episodes
            if e.q_proposed_visit_count_log
        ]
        q_proposed_value = [
            e.mean_q_proposed_value()
            for e in self.episodes
            if e.q_proposed_value_log
        ]
        compositionality_error = [
            e.mean_compositionality_error()
            for e in self.episodes
            if e.compositionality_error_log
        ]
        compositionality_holds = [
            e.compositionality_holds_rate()
            for e in self.episodes
            if e.compositionality_holds_flags
        ]
        compositionality_median_error = [
            e.median_compositionality_error()
            for e in self.episodes
            if e.compositionality_error_log
        ]
        compositionality_p95_error = [
            e.p95_compositionality_error()
            for e in self.episodes
            if e.compositionality_error_log
        ]
        compositionality_max_error = [
            e.max_compositionality_error()
            for e in self.episodes
            if e.compositionality_error_log
        ]
        compositionality_goal_error = [
            e.mean_compositionality_goal_error()
            for e in self.episodes
            if e.compositionality_goal_error_log
        ]
        compositionality_modulator_error = [
            e.mean_compositionality_modulator_error()
            for e in self.episodes
            if e.compositionality_modulator_error_log
        ]
        compositionality_action_holds = [
            e.compositionality_action_holds_rate()
            for e in self.episodes
            if e.compositionality_action_holds_flags
        ]

        result = {
            "label":            self.label,
            "n_episodes":       n,
            "completion_rate":  self._stat(cr),
            "lava_rate":        self._stat(lr),
            "total_reward":     self._stat(tr),
            "srv_rate":         self._stat(srv),
            "unsafe_rate":      self._stat(unsafe),
            "recovery_time":    self._stat(rt),
            "final_energy":     self._stat(final_energy),
            "survival_rate":    self._stat(survival),
            "path_efficiency":  self._stat(path_efficiency),
            "minerals_collected": self._stat(minerals_collected),
        }

        if mot_srv:
            result["mot_srv_rate"] = self._stat(mot_srv)
        if env_srv:
            result["env_srv_rate"] = self._stat(env_srv)
        if mot_boundary:
            result["mot_boundary_rate"] = self._stat(mot_boundary)
        if mot_pressure:
            result["mot_pressure"] = self._stat(mot_pressure)
        if mot_boundary_rt:
            result["mot_boundary_recovery_time"] = self._stat(mot_boundary_rt)
        if appraisal_influence:
            result["appraisal_influence_rate"] = self._stat(appraisal_influence)
        if appraisal_score_shift:
            result["appraisal_score_shift"] = self._stat(appraisal_score_shift)
        if exploration:
            result["exploration_rate"] = self._stat(exploration)
        if risk_penalty:
            result["risk_penalty"] = self._stat(risk_penalty)
        if selector_influence:
            result["selector_influence_rate"] = self._stat(selector_influence)
        if hard_safety_intervention:
            result["hard_safety_intervention_rate"] = self._stat(
                hard_safety_intervention
            )
        if q_regret:
            result["q_regret"] = self._stat(q_regret)
        if task_score:
            result["task_score"] = self._stat(task_score)
        if safety_score:
            result["safety_score"] = self._stat(safety_score)
        if task_safety_disagreement:
            result["task_safety_disagreement"] = self._stat(
                task_safety_disagreement
            )
        if q_shortlist_size:
            result["q_shortlist_size"] = self._stat(q_shortlist_size)
        if q_proposed_lava:
            result["q_proposed_lava_rate"] = self._stat(q_proposed_lava)
        if executed_lava_entry:
            result["executed_lava_entry_rate"] = self._stat(
                executed_lava_entry
            )
        if q_proposal_blocked:
            result["q_proposal_blocked_rate"] = self._stat(q_proposal_blocked)
        if executed_changed_q:
            result["executed_changed_q_rate"] = self._stat(executed_changed_q)
        if q_proposed_risk:
            result["q_proposed_risk"] = self._stat(q_proposed_risk)
        if executed_risk:
            result["executed_risk"] = self._stat(executed_risk)
        if q_proposed_visit_count:
            result["q_proposed_visit_count"] = self._stat(
                q_proposed_visit_count
            )
        if q_proposed_value:
            result["q_proposed_value"] = self._stat(q_proposed_value)
        if compositionality_error:
            result["compositionality_error"] = self._stat(
                compositionality_error
            )
        if compositionality_holds:
            result["compositionality_holds_rate"] = self._stat(
                compositionality_holds
            )
        if compositionality_median_error:
            result["compositionality_median_error"] = self._stat(
                compositionality_median_error
            )
        if compositionality_p95_error:
            result["compositionality_p95_error"] = self._stat(
                compositionality_p95_error
            )
        if compositionality_max_error:
            result["compositionality_max_error"] = self._stat(
                compositionality_max_error
            )
        if compositionality_goal_error:
            result["compositionality_goal_error"] = self._stat(
                compositionality_goal_error
            )
        if compositionality_modulator_error:
            result["compositionality_modulator_error"] = self._stat(
                compositionality_modulator_error
            )
        if compositionality_action_holds:
            result["compositionality_action_holds_rate"] = self._stat(
                compositionality_action_holds
            )

        return result
