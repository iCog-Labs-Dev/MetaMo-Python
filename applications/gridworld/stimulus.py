from dataclasses import dataclass


@dataclass(frozen=True)
class GridWorldStimulus:
    """
    Raw application stimulus for the GridWorld application.

    Each field has a separate task meaning and is normalized to [0, 1].
    Keeping the signals operationally distinct avoids representing the same
    underlying distance measurement under several aliases.
    """

    goal_proximity: float
    hazard_pressure: float
    energy_deficit: float
    time_pressure: float
    safe_progress: float
    safe_mobility: float
