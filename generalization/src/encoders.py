from __future__ import annotations

"""State encoders for delivery generalization experiments."""

from collections import deque
from functools import lru_cache
from typing import Dict, List

import numpy as np

from src.env import GridDeliveryEnv


def _resolve_current_target(env: GridDeliveryEnv) -> tuple[tuple[int, int], str]:
    """Return the current navigation target and its semantic mode."""

    if env.active_task >= 0:
        return env.instance.delivery_tasks[env.active_task].dropoff, "dropoff"

    for task_idx, task in enumerate(env.instance.delivery_tasks):
        delivered = bool(env.delivered_mask & (1 << task_idx))
        if not delivered:
            return task.pickup, "pickup"
    return env.instance.goal, "goal"


@lru_cache(maxsize=1024)
def _distance_map_to_target(
    rows: int,
    cols: int,
    obstacles: tuple[tuple[int, int], ...],
    target: tuple[int, int],
) -> tuple[int, ...]:
    """Compute a BFS distance map from every free cell to the target."""

    distances = [-1] * (rows * cols)
    blocked = set(obstacles)
    if target in blocked:
        return tuple(distances)

    queue = deque([target])
    distances[target[0] * cols + target[1]] = 0

    while queue:
        row, col = queue.popleft()
        base_distance = distances[row * cols + col]
        for next_row, next_col in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if not (0 <= next_row < rows and 0 <= next_col < cols):
                continue
            if (next_row, next_col) in blocked:
                continue
            flat_index = next_row * cols + next_col
            if distances[flat_index] >= 0:
                continue
            distances[flat_index] = int(base_distance) + 1
            queue.append((next_row, next_col))

    return tuple(distances)


def _lookup_distance(distance_map: tuple[int, ...], cols: int, position: tuple[int, int]) -> int | None:
    value = int(distance_map[position[0] * cols + position[1]])
    return None if value < 0 else value


def _clip_nonnegative(value: int | None, limit: int) -> int:
    if value is None:
        return int(limit + 1)
    return int(min(value, limit))


def _encode_distance_delta(current_distance: int | None, neighbor_distance: int | None, blocked: bool) -> int:
    """Bucket distance changes so similar map situations share one tabular state."""

    if blocked:
        return 9
    if current_distance is None or neighbor_distance is None:
        return 8
    return int(max(-2, min(4, neighbor_distance - current_distance)))


def encode_absolute_state(env: GridDeliveryEnv) -> tuple[int, int, int, int]:
    """Return the tabular absolute state directly."""

    return env.state()


def encode_feature_state(env: GridDeliveryEnv) -> Dict[str, int]:
    """Return a compact feature representation for generalized Q-learning."""

    row, col = env.position
    mask = env.action_mask()
    target, target_mode = _resolve_current_target(env)

    return {
        "row": int(row),
        "col": int(col),
        "wall_up": int(mask[0] == 0),
        "wall_down": int(mask[1] == 0),
        "wall_left": int(mask[2] == 0),
        "wall_right": int(mask[3] == 0),
        "target_dr": int(target[0] - row),
        "target_dc": int(target[1] - col),
        "target_mode_pickup": int(target_mode == "pickup"),
        "target_mode_dropoff": int(target_mode == "dropoff"),
        "target_mode_goal": int(target_mode == "goal"),
        "active_task": int(env.active_task),
        "delivered_mask": int(env.delivered_mask),
        "task_count": int(env.instance.task_count),
    }


def encode_generalized_feature_state(env: GridDeliveryEnv) -> Dict[str, int]:
    """Return a stronger hand-designed representation for cross-map transfer.

    This version removes absolute coordinates and emphasizes local structure plus
    target-relative geometry.
    """

    row, col = env.position
    mask = env.action_mask()
    obstacle_set = env.instance.obstacle_set
    target, target_mode = _resolve_current_target(env)

    target_dr = int(target[0] - row)
    target_dc = int(target[1] - col)

    def blocked(position: tuple[int, int]) -> int:
        r, c = position
        return int((r < 0) or (r >= env.rows) or (c < 0) or (c >= env.cols) or (position in obstacle_set))

    up = int(mask[0] == 1)
    down = int(mask[1] == 1)
    left = int(mask[2] == 1)
    right = int(mask[3] == 1)

    return {
        "wall_up": int(mask[0] == 0),
        "wall_down": int(mask[1] == 0),
        "wall_left": int(mask[2] == 0),
        "wall_right": int(mask[3] == 0),
        "corridor_vertical": int(up and down and (not left) and (not right)),
        "corridor_horizontal": int(left and right and (not up) and (not down)),
        "dead_end": int((up + down + left + right) == 1),
        "junction_3plus": int((up + down + left + right) >= 3),
        "diag_ul_blocked": blocked((row - 1, col - 1)),
        "diag_ur_blocked": blocked((row - 1, col + 1)),
        "diag_dl_blocked": blocked((row + 1, col - 1)),
        "diag_dr_blocked": blocked((row + 1, col + 1)),
        "target_dr_sign": int(np.sign(target_dr)),
        "target_dc_sign": int(np.sign(target_dc)),
        "target_dr_abs_clip": int(min(abs(target_dr), 4)),
        "target_dc_abs_clip": int(min(abs(target_dc), 4)),
        "target_manhattan_clip": int(min(abs(target_dr) + abs(target_dc), 8)),
        "moving_toward_up": int(target_dr < 0),
        "moving_toward_down": int(target_dr > 0),
        "moving_toward_left": int(target_dc < 0),
        "moving_toward_right": int(target_dc > 0),
        "target_mode_pickup": int(target_mode == "pickup"),
        "target_mode_dropoff": int(target_mode == "dropoff"),
        "target_mode_goal": int(target_mode == "goal"),
        "active_task": int(env.active_task),
        "delivered_mask": int(env.delivered_mask),
        "task_count": int(env.instance.task_count),
    }


def encode_path_distance_state(env: GridDeliveryEnv) -> Dict[str, int]:
    """Encode global path-distance structure rather than raw coordinates.

    This representation uses shortest-path distance maps on the observed grid to
    summarize how each action changes progress toward the current target.
    """

    row, col = env.position
    target, target_mode = _resolve_current_target(env)
    goal = env.instance.goal
    obstacle_key = tuple(env.instance.obstacles)

    target_distance_map = _distance_map_to_target(env.rows, env.cols, obstacle_key, target)
    goal_distance_map = _distance_map_to_target(env.rows, env.cols, obstacle_key, goal)

    current_target_distance = _lookup_distance(target_distance_map, env.cols, env.position)
    current_goal_distance = _lookup_distance(goal_distance_map, env.cols, env.position)

    best_neighbor_target_distance: int | None = None
    per_action_target_distances: Dict[str, int | None] = {}
    per_action_blocked: Dict[str, bool] = {}

    for action, (dr, dc) in env.ACTIONS.items():
        next_position = (row + dr, col + dc)
        action_name = env.ACTION_NAMES[action]
        blocked = (not env.in_bounds(next_position)) or env.is_blocked(next_position)
        per_action_blocked[action_name] = bool(blocked)
        if blocked:
            per_action_target_distances[action_name] = None
            continue

        next_distance = _lookup_distance(target_distance_map, env.cols, next_position)
        per_action_target_distances[action_name] = next_distance
        if next_distance is None:
            continue
        if best_neighbor_target_distance is None or next_distance < best_neighbor_target_distance:
            best_neighbor_target_distance = next_distance

    features: Dict[str, int] = {
        "target_mode_pickup": int(target_mode == "pickup"),
        "target_mode_dropoff": int(target_mode == "dropoff"),
        "target_mode_goal": int(target_mode == "goal"),
        "active_task": int(env.active_task),
        "delivered_mask": int(env.delivered_mask),
        "task_count": int(env.instance.task_count),
        "target_sp_clip": _clip_nonnegative(current_target_distance, limit=12),
        "goal_sp_clip": _clip_nonnegative(current_goal_distance, limit=12),
        "target_dr_sign": int(np.sign(target[0] - row)),
        "target_dc_sign": int(np.sign(target[1] - col)),
        "goal_dr_sign": int(np.sign(goal[0] - row)),
        "goal_dc_sign": int(np.sign(goal[1] - col)),
        "best_target_move_count": int(
            sum(
                1
                for value in per_action_target_distances.values()
                if value is not None and value == best_neighbor_target_distance
            )
        ),
        "at_target": int(env.position == target),
        "at_goal": int(env.position == goal),
    }

    for action, _ in env.ACTIONS.items():
        action_name = env.ACTION_NAMES[action]
        blocked = per_action_blocked[action_name]
        next_target_distance = per_action_target_distances[action_name]
        features[f"{action_name}_blocked"] = int(blocked)
        features[f"{action_name}_target_delta"] = _encode_distance_delta(
            current_target_distance,
            next_target_distance,
            blocked,
        )
        features[f"{action_name}_best_to_target"] = int(
            (not blocked)
            and next_target_distance is not None
            and best_neighbor_target_distance is not None
            and next_target_distance == best_neighbor_target_distance
        )

    return features


def encode_patch_state(env: GridDeliveryEnv, radius: int = 2) -> Dict[str, object]:
    """Return a local map patch plus task-progress vector.

    This is a bridge representation for future approximate / deep Q-learning.
    """

    row, col = env.position
    side = 2 * radius + 1
    patch = np.full((side, side), fill_value=-1, dtype=np.int32)
    obstacle_set = env.instance.obstacle_set

    for patch_r, world_r in enumerate(range(row - radius, row + radius + 1)):
        for patch_c, world_c in enumerate(range(col - radius, col + radius + 1)):
            position = (world_r, world_c)
            if not (0 <= world_r < env.rows and 0 <= world_c < env.cols):
                patch[patch_r, patch_c] = -1
            elif position in obstacle_set:
                patch[patch_r, patch_c] = 1
            else:
                patch[patch_r, patch_c] = 0

    features = encode_feature_state(env)
    return {
        "patch": patch.tolist(),
        "task_features": features,
    }


def encode_patch3x3_state(env: GridDeliveryEnv) -> Dict[str, object]:
    """Return a 3x3 local observation window plus compact task metadata."""

    local_patch = encode_patch_state(env, radius=1)["patch"]
    target, target_mode = _resolve_current_target(env)
    row, col = env.position
    return {
        "patch3x3": local_patch,
        "target_dr_sign": int(np.sign(target[0] - row)),
        "target_dc_sign": int(np.sign(target[1] - col)),
        "target_mode_pickup": int(target_mode == "pickup"),
        "target_mode_dropoff": int(target_mode == "dropoff"),
        "target_mode_goal": int(target_mode == "goal"),
        "active_task": int(env.active_task),
        "delivered_mask": int(env.delivered_mask),
        "task_count": int(env.instance.task_count),
    }


def encode_patch3x3_plus_state(env: GridDeliveryEnv) -> Dict[str, object]:
    """Return a richer 3x3 local-window state with navigation context.

    This keeps the raw 3x3 observation but augments it with:
    - shortest-path progress signals
    - action-level target-distance hints
    - compact local topology summaries
    """

    local_patch = encode_patch_state(env, radius=1)["patch"]
    patch_array = np.asarray(local_patch, dtype=np.int32)
    distance_features = encode_path_distance_state(env)
    topology_features = encode_generalized_feature_state(env)

    selected_topology = {
        "corridor_vertical": int(topology_features["corridor_vertical"]),
        "corridor_horizontal": int(topology_features["corridor_horizontal"]),
        "dead_end": int(topology_features["dead_end"]),
        "junction_3plus": int(topology_features["junction_3plus"]),
        "diag_ul_blocked": int(topology_features["diag_ul_blocked"]),
        "diag_ur_blocked": int(topology_features["diag_ur_blocked"]),
        "diag_dl_blocked": int(topology_features["diag_dl_blocked"]),
        "diag_dr_blocked": int(topology_features["diag_dr_blocked"]),
    }

    return {
        "patch3x3": local_patch,
        "patch_free_count": int(np.sum(patch_array == 0)),
        "patch_obstacle_count": int(np.sum(patch_array == 1)),
        "patch_boundary_count": int(np.sum(patch_array == -1)),
        **selected_topology,
        **distance_features,
    }


def encode_patch3x3_plus_vector(env: GridDeliveryEnv) -> np.ndarray:
    """Vectorize the enhanced 3x3 local-window state for linear approximation."""

    state = encode_patch3x3_plus_state(env)
    patch = np.asarray(state["patch3x3"], dtype=np.float64).reshape(-1)

    scalar_feature_order = [
        "patch_free_count",
        "patch_obstacle_count",
        "patch_boundary_count",
        "corridor_vertical",
        "corridor_horizontal",
        "dead_end",
        "junction_3plus",
        "diag_ul_blocked",
        "diag_ur_blocked",
        "diag_dl_blocked",
        "diag_dr_blocked",
        "target_mode_pickup",
        "target_mode_dropoff",
        "target_mode_goal",
        "active_task",
        "delivered_mask",
        "task_count",
        "target_sp_clip",
        "goal_sp_clip",
        "target_dr_sign",
        "target_dc_sign",
        "goal_dr_sign",
        "goal_dc_sign",
        "best_target_move_count",
        "at_target",
        "at_goal",
        "up_blocked",
        "up_target_delta",
        "up_best_to_target",
        "down_blocked",
        "down_target_delta",
        "down_best_to_target",
        "left_blocked",
        "left_target_delta",
        "left_best_to_target",
        "right_blocked",
        "right_target_delta",
        "right_best_to_target",
    ]
    scalar_values = np.asarray([float(state[key]) for key in scalar_feature_order], dtype=np.float64)
    return np.concatenate([np.array([1.0], dtype=np.float64), patch, scalar_values], axis=0)
