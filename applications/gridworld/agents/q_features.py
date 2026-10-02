"""Shared GridWorld Q-learning feature helpers."""

from __future__ import annotations

from applications.gridworld.config import ACTION_DELTAS

ACTION_COUNT = len(ACTION_DELTAS)


def sign(value: int) -> int:
    """Return the sign of an integer as -1, 0, or 1."""
    return (value > 0) - (value < 0)


def manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Compute Manhattan distance between two grid cells."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def in_bounds(pos: tuple[int, int], grid_size: int = 10) -> bool:
    """Return whether a grid cell lies inside the square GridWorld."""
    return 0 <= pos[0] < grid_size and 0 <= pos[1] < grid_size


def next_position(
    pos: tuple[int, int],
    action: int,
    grid_size: int = 10,
) -> tuple[tuple[int, int], bool]:
    """
    Project a movement action.

    Returns the projected position and whether the action would hit the
    boundary. Boundary actions stay in place, matching the environment.
    """
    dr, dc = ACTION_DELTAS[action]
    candidate = (pos[0] + dr, pos[1] + dc)
    if in_bounds(candidate, grid_size):
        return candidate, False
    return pos, True


def valid_actions(
    state: dict,
    grid_size: int = 10,
    avoid_lava: bool = False,
    allow_boundary: bool = False,
) -> list[int]:
    """
    Return actions permitted by the shared comparison policy.

    When lava avoidance filters every in-bounds action, the function falls back
    to in-bounds actions so agents are not trapped by the filter itself.
    """
    pos = state["pos"]
    lava_cells = set(state.get("lava_cells", ()))
    in_bounds_actions: list[int] = []
    safe_actions: list[int] = []

    for action in range(ACTION_COUNT):
        projected, hit_boundary = next_position(pos, action, grid_size)
        if hit_boundary and not allow_boundary:
            continue
        in_bounds_actions.append(action)
        if not avoid_lava or projected not in lava_cells:
            safe_actions.append(action)

    if safe_actions:
        return safe_actions
    if in_bounds_actions:
        return in_bounds_actions
    return list(range(ACTION_COUNT))


def encode_absolute_task(state: dict) -> tuple[int, int, int, int]:
    """Encode absolute agent and mineral positions."""
    ar, ac = state["pos"]
    mr, mc = state["mineral_pos"]
    return (ar, ac, mr, mc)


def encode_compact_hazard(
    state: dict,
    grid_size: int = 10,
    max_distance_bin: int = 4,
) -> tuple[int, int, int, int, int, int, int, int]:
    """
    Encode relative task and local hazard structure.

    This matches the current baseline representation and is shared so MetaMo
    can be evaluated with the same Q-learning information.
    """
    ar, ac = state["pos"]
    mr, mc = state["mineral_pos"]
    lava_cells = set(state.get("lava_cells", ()))

    dy = mr - ar
    dx = mc - ac
    local_lava_mask = 0
    boundary_mask = 0

    for action in range(ACTION_COUNT):
        projected, hit_boundary = next_position((ar, ac), action, grid_size)
        if hit_boundary:
            boundary_mask |= 1 << action
        if projected in lava_cells:
            local_lava_mask |= 1 << action

    lava_distance = int(state.get("lava_distance", grid_size * 2))
    lava_distance_bin = min(lava_distance, max_distance_bin + 1)

    return (
        sign(dy),
        sign(dx),
        min(abs(dy), max_distance_bin),
        min(abs(dx), max_distance_bin),
        local_lava_mask,
        boundary_mask,
        lava_distance_bin,
        int(state.get("in_lava", False)),
    )


def encode_state(
    state: dict,
    mode: str = "compact_hazard",
    grid_size: int = 10,
    max_distance_bin: int = 4,
) -> tuple:
    """Encode a GridWorld state under a named Q-feature mode."""
    if mode == "absolute_task":
        return encode_absolute_task(state)
    if mode == "compact_hazard":
        return encode_compact_hazard(
            state,
            grid_size=grid_size,
            max_distance_bin=max_distance_bin,
        )
    raise ValueError(f"unknown Q encoder mode: {mode}")
