from core.schema import MagusGoalSchema, ModulatorSchema, MotivationSchema


GRIDWORLD_SCHEMA = MotivationSchema(
    goals=MagusGoalSchema(
        primary_goals=(
            "mineral_acquisition",
            "energy_preservation",
            "navigation_efficiency",
        ),
        anti_goals=(
            "lava_exposure",
            "boundary_collision",
        ),
    ),
    modulators=ModulatorSchema(),
    decision_profile_name="gridworld_magus",
    appraisal_profile_name="gridworld_openpsi",
)
