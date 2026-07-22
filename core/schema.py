from dataclasses import dataclass


OPENPSI_MODULATORS = (
    "valence",
    "arousal",
    "approach",
    "resolution",
    "threshold",
    "securing",
)


def _validate_unique_names(names: tuple[str, ...], label: str) -> None:
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError(f"{label} names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} names must be unique")


def _normalize_names(names: tuple[str, ...], label: str) -> tuple[str, ...]:
    """
    Normalize caller-provided schema names into immutable string tuples.
    """
    if names is None:
        raise ValueError(f"{label} names must be provided")
    return tuple(name.strip() if isinstance(name, str) else name for name in names)


@dataclass(frozen=True)
class MagusGoalSchema:
    """
    MAGUS/OEI goal layout:
    (g_over^Ind, g_over^Trans, g_1, ..., g_P, a_1, ..., a_Q).
    """

    primary_goals: tuple[str, ...]
    anti_goals: tuple[str, ...] = ()
    individuation_name: str = "individuation"
    transcendence_name: str = "transcendence"

    def __post_init__(self):
        object.__setattr__(self, "primary_goals", _normalize_names(self.primary_goals, "primary goal"))
        object.__setattr__(self, "anti_goals", _normalize_names(self.anti_goals, "anti-goal"))
        overgoals = _normalize_names(
            (self.individuation_name, self.transcendence_name),
            "overgoal",
        )
        object.__setattr__(self, "individuation_name", overgoals[0])
        object.__setattr__(self, "transcendence_name", overgoals[1])
        _validate_unique_names(self.primary_goals, "primary goal")
        _validate_unique_names(self.anti_goals, "anti-goal")
        _validate_unique_names(
            (self.individuation_name, self.transcendence_name),
            "overgoal",
        )
        _validate_unique_names(self.goal_names, "goal")

    @property
    def goal_names(self) -> tuple[str, ...]:
        return (
            self.individuation_name,
            self.transcendence_name,
            *self.primary_goals,
            *self.anti_goals,
        )

    @property
    def num_goals(self) -> int:
        return len(self.goal_names)

    @property
    def primary_start(self) -> int:
        return 2

    @property
    def anti_goal_start(self) -> int:
        return 2 + len(self.primary_goals)

    def index(self, name: str) -> int:
        try:
            return self.goal_names.index(name)
        except ValueError as error:
            raise KeyError(f"unknown goal name: {name}") from error

    def is_primary_goal(self, name: str) -> bool:
        return name in self.primary_goals

    def is_anti_goal(self, name: str) -> bool:
        return name in self.anti_goals


@dataclass(frozen=True)
class ModulatorSchema:
    """
    Modulator layout: OpenPsi modulators followed by application-specific modulators.
    """

    application_specific: tuple[str, ...] = ()
    openpsi_modulators: tuple[str, ...] = OPENPSI_MODULATORS

    def __post_init__(self):
        object.__setattr__(self, "openpsi_modulators", _normalize_names(self.openpsi_modulators, "core OpenPsi modulator"))
        object.__setattr__(
            self,
            "application_specific",
            _normalize_names(self.application_specific, "application-specific modulator"),
        )
        _validate_unique_names(self.openpsi_modulators, "core OpenPsi modulator")
        _validate_unique_names(self.application_specific, "application-specific modulator")
        _validate_unique_names(self.modulator_names, "modulator")

    @property
    def modulator_names(self) -> tuple[str, ...]:
        return (*self.openpsi_modulators, *self.application_specific)

    @property
    def num_modulators(self) -> int:
        return len(self.modulator_names)

    @property
    def core_count(self) -> int:
        return len(self.openpsi_modulators)

    def index(self, name: str) -> int:
        try:
            return self.modulator_names.index(name)
        except ValueError as error:
            raise KeyError(f"unknown modulator name: {name}") from error

    def is_openpsi_modulators(self, name: str) -> bool:
        return name in self.openpsi_modulators

    def is_application_specific(self, name: str) -> bool:
        return name in self.application_specific


@dataclass(frozen=True)
class MotivationSchema:
    """
    Complete coordinate schema for a MetaMo motivational state.
    """

    goals: MagusGoalSchema
    modulators: ModulatorSchema
    decision_profile_name: str = "magus_additive_default"
    appraisal_profile_name: str = "openpsi_default"

    @property
    def num_goals(self) -> int:
        return self.goals.num_goals

    @property
    def num_modulators(self) -> int:
        return self.modulators.num_modulators

    @property
    def goal_names(self) -> tuple[str, ...]:
        return self.goals.goal_names

    @property
    def modulator_names(self) -> tuple[str, ...]:
        return self.modulators.modulator_names

    def goal_index(self, name: str) -> int:
        return self.goals.index(name)

    def modulator_index(self, name: str) -> int:
        return self.modulators.index(name)

    def is_anti_goal(self, name: str) -> bool:
        return self.goals.is_anti_goal(name)


DEFAULT_MOTIVATION_SCHEMA = MotivationSchema(
    goals=MagusGoalSchema(
        primary_goals=(
            "help",
            "curiosity",
            "novelty",
            "self_improvement",
            "ethics",
            "sociality",
        ),
    ),
    modulators=ModulatorSchema(),
)
