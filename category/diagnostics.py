from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class MetaMoDiagnostics:
    """
    Telemetry for one MetaMo transition.
    """

    action_id: str
    lax_error: float
    lax_tolerance: float
    lax_holds: bool
    contractive_holds: bool
    target_in_safe_region: bool
    final_in_safe_region: bool
    boundary_pressure_before: float
    boundary_pressure_target: float
    boundary_pressure_final: float
    projection_delta: float
    target_distance: float
    state_drift: float
    self_model_drift: float
    combined_self_model_drift: float
    self_model_drift_tolerance: float
    self_model_drift_holds: bool
    blend_alpha: float
    base_blend_alpha: float


@dataclass(frozen=True)
class MetaMoDiagnosticsSummary:
    """
    Aggregate telemetry over a run.
    """

    count: int
    max_lax_error: float
    mean_lax_error: float
    max_projection_delta: float
    mean_projection_delta: float
    max_boundary_pressure: float
    mean_boundary_pressure: float
    max_self_model_drift: float
    mean_self_model_drift: float
    max_state_drift: float
    mean_state_drift: float
    safe_region_violation_rate: float
    law_violation_rate: float


class MetaMoDiagnosticsHistory:
    """
    Persistent diagnostics collector for MetaMo transitions.
    """

    def __init__(self):
        self.records: List[MetaMoDiagnostics] = []

    def append(self, diagnostics: MetaMoDiagnostics) -> None:
        self.records.append(diagnostics)

    def extend(self, diagnostics: List[MetaMoDiagnostics]) -> None:
        self.records.extend(diagnostics)

    def clear(self) -> None:
        self.records.clear()

    def __len__(self) -> int:
        return len(self.records)

    def last(self) -> MetaMoDiagnostics | None:
        if not self.records:
            return None
        return self.records[-1]

    def to_rows(self) -> List[dict]:
        return [record.__dict__.copy() for record in self.records]

    def write_csv(self, path: str) -> None:
        """
        Persist diagnostics history to a CSV file.
        """
        import csv

        rows = self.to_rows()
        if not rows:
            return

        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def summary(self) -> MetaMoDiagnosticsSummary:
        if not self.records:
            return MetaMoDiagnosticsSummary(
                count=0,
                max_lax_error=0.0,
                mean_lax_error=0.0,
                max_projection_delta=0.0,
                mean_projection_delta=0.0,
                max_boundary_pressure=0.0,
                mean_boundary_pressure=0.0,
                max_self_model_drift=0.0,
                mean_self_model_drift=0.0,
                max_state_drift=0.0,
                mean_state_drift=0.0,
                safe_region_violation_rate=0.0,
                law_violation_rate=0.0,
            )

        lax_errors = np.array([r.lax_error for r in self.records], dtype=float)
        projection_deltas = np.array([r.projection_delta for r in self.records], dtype=float)
        boundary_pressures = np.array([r.boundary_pressure_final for r in self.records], dtype=float)
        self_model_drifts = np.array([r.self_model_drift for r in self.records], dtype=float)
        state_drifts = np.array([r.state_drift for r in self.records], dtype=float)
        safe_violations = np.array([not r.final_in_safe_region for r in self.records], dtype=float)
        law_violations = np.array(
            [
                (not r.lax_holds)
                or (not r.contractive_holds)
                or (not r.self_model_drift_holds)
                for r in self.records
            ],
            dtype=float,
        )

        return MetaMoDiagnosticsSummary(
            count=len(self.records),
            max_lax_error=float(np.max(lax_errors)),
            mean_lax_error=float(np.mean(lax_errors)),
            max_projection_delta=float(np.max(projection_deltas)),
            mean_projection_delta=float(np.mean(projection_deltas)),
            max_boundary_pressure=float(np.max(boundary_pressures)),
            mean_boundary_pressure=float(np.mean(boundary_pressures)),
            max_self_model_drift=float(np.max(self_model_drifts)),
            mean_self_model_drift=float(np.mean(self_model_drifts)),
            max_state_drift=float(np.max(state_drifts)),
            mean_state_drift=float(np.mean(state_drifts)),
            safe_region_violation_rate=float(np.mean(safe_violations)),
            law_violation_rate=float(np.mean(law_violations)),
        )
