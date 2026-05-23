"""Success and failure termination terms for SO-101 Bench."""

from __future__ import annotations

import math

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from so101_bench.benchmark import (
    BETWEEN_LINE_TOLERANCE_M,
    BIN_DISPLACEMENT_LIMIT_M,
    BOUNDARY_DISPLACEMENT_LIMIT_M,
    DIRECTIONS,
    GRASP_ATTEMPT_OBJECT_DISTANCE_M,
    NON_TARGET_DISPLACEMENT_LIMIT_M,
    ON_TOP_VERTICAL_TOLERANCE_M,
    SPATIAL_SUCCESS_DISTANCE_M,
    TASK_BETWEEN,
    TASK_BIN,
    TASK_MOVE,
    TASK_NEXT_TO,
)

from .resets import benchmark_object_positions, mark_benchmark_robot_start

FAILURE_REASON_NONE = "none"
FAILURE_REASON_MAX_GRASP_ATTEMPTS = "max_grasp_attempts"
FAILURE_REASON_BIN_DISPLACED = "bin_displaced"
FAILURE_REASON_NON_TARGET_MOVED = "non_target_moved"
FAILURE_REASON_MOVE_BOUNDARY_MOVED = "move_boundary_moved"
FAILURE_REASON_MOVE_PAST_BOUNDARY = "move_past_boundary"
FAILURE_REASON_TARGET_ON_TOP = "target_on_top"

DEFAULT_SUCCESS_CONFIRM_TIME_S = 0.25


def _active_mask(env: ManagerBasedRLEnv, object_asset_names: list[str]) -> torch.Tensor:
    if hasattr(env, "_so101_active_object_mask"):
        return env._so101_active_object_mask
    return torch.ones((env.num_envs, len(object_asset_names)), dtype=torch.bool, device=env.device)


def _object_positions(env: ManagerBasedRLEnv, object_asset_names: list[str]) -> torch.Tensor:
    return benchmark_object_positions(env, object_asset_names)


def _object_half_extents(env: ManagerBasedRLEnv, object_asset_names: list[str]) -> torch.Tensor:
    if hasattr(env, "_so101_object_half_extents"):
        return env._so101_object_half_extents
    fallback = torch.full(
        (env.num_envs, len(object_asset_names), 3),
        0.02,
        dtype=torch.float32,
        device=env.device,
    )
    return fallback


def _bin_half_extents(env: ManagerBasedRLEnv) -> torch.Tensor:
    if hasattr(env, "_so101_bin_half_extents"):
        return env._so101_bin_half_extents
    return torch.tensor((0.125, 0.095, 0.08), dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)


def _task_is(env: ManagerBasedRLEnv, task_family: str) -> torch.Tensor:
    if not hasattr(env, "_so101_task_family"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return torch.tensor([task == task_family for task in env._so101_task_family], dtype=torch.bool, device=env.device)


def _target_indices(env: ManagerBasedRLEnv) -> torch.Tensor:
    if hasattr(env, "_so101_target_object_ids"):
        return env._so101_target_object_ids
    return torch.zeros(env.num_envs, dtype=torch.long, device=env.device)


def _referent_indices(env: ManagerBasedRLEnv) -> torch.Tensor:
    if hasattr(env, "_so101_referent_object_ids"):
        return env._so101_referent_object_ids
    return torch.zeros((env.num_envs, 2), dtype=torch.long, device=env.device)


def _direction_indices(env: ManagerBasedRLEnv) -> torch.Tensor:
    if hasattr(env, "_so101_direction_ids"):
        return env._so101_direction_ids
    return torch.zeros(env.num_envs, dtype=torch.long, device=env.device)


def _gather_by_index(values: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return values[torch.arange(values.shape[0], device=values.device), ids]


def _aabb_surface_distance_xy(
    first_pos: torch.Tensor,
    first_half_extents: torch.Tensor,
    second_pos: torch.Tensor,
    second_half_extents: torch.Tensor,
) -> torch.Tensor:
    separation = torch.abs(first_pos[..., :2] - second_pos[..., :2])
    separation -= first_half_extents[..., :2] + second_half_extents[..., :2]
    return torch.linalg.vector_norm(torch.clamp(separation, min=0.0), dim=-1)


def _target_is_on_top(env: ManagerBasedRLEnv, object_asset_names: list[str]) -> torch.Tensor:
    positions = _object_positions(env, object_asset_names)
    half_extents = _object_half_extents(env, object_asset_names)
    active = _active_mask(env, object_asset_names)
    target_ids = _target_indices(env)

    target_pos = _gather_by_index(positions, target_ids).unsqueeze(1)
    target_half = _gather_by_index(half_extents, target_ids).unsqueeze(1)
    xy_overlap = _aabb_surface_distance_xy(target_pos, target_half, positions, half_extents) <= 0.002

    target_bottom = target_pos[..., 2] - target_half[..., 2]
    object_top = positions[..., 2] + half_extents[..., 2]
    supported_by_object = torch.abs(target_bottom - object_top) <= ON_TOP_VERTICAL_TOLERANCE_M
    target_above_object = target_pos[..., 2] > positions[..., 2] + 0.5 * half_extents[..., 2]

    target_mask = torch.zeros_like(active)
    target_mask[torch.arange(env.num_envs, device=env.device), target_ids] = True
    return torch.any(xy_overlap & supported_by_object & target_above_object & active & (~target_mask), dim=1)


def _attempt_object_mask(env: ManagerBasedRLEnv, object_asset_names: list[str]) -> torch.Tensor:
    active = _active_mask(env, object_asset_names)
    target_mask = torch.zeros_like(active)
    target_mask[torch.arange(env.num_envs, device=env.device), _target_indices(env)] = True
    return torch.where(_task_is(env, TASK_BIN).unsqueeze(1), active, active & target_mask)


def _env_step_dt(env: ManagerBasedRLEnv) -> float:
    step_dt = getattr(env, "step_dt", env.cfg.sim.dt * env.cfg.decimation)
    return float(step_dt)


def _episode_age_s(env: ManagerBasedRLEnv) -> torch.Tensor:
    step_dt = _env_step_dt(env)
    return env.episode_length_buf.to(dtype=torch.float32) * step_dt


def _episode_age_at_least(env: ManagerBasedRLEnv, seconds: float) -> torch.Tensor:
    return _episode_age_s(env) >= seconds


def _confirmation_steps(
    env: ManagerBasedRLEnv,
    confirm_time_s: float,
    confirm_steps: int | None = None,
) -> int:
    if confirm_steps is not None:
        return confirm_steps
    return max(1, math.ceil(confirm_time_s / _env_step_dt(env)))


def bin_success(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    bin_name: str,
    bin_inner_half_extents: tuple[float, float, float] = (0.108, 0.078, 0.078),
    confirm_steps: int | None = None,
    confirm_time_s: float = DEFAULT_SUCCESS_CONFIRM_TIME_S,
) -> torch.Tensor:
    """Success when all active objects are contained in the plastic bin."""

    object_pos_w = _object_positions(env, object_asset_names)
    active = _active_mask(env, object_asset_names)
    bin_asset: RigidObject = env.scene[bin_name]
    bin_pos_w = bin_asset.data.root_pos_w
    bin_quat_inv = math_utils.quat_inv(bin_asset.data.root_quat_w)

    rel = object_pos_w - bin_pos_w.unsqueeze(1)
    rel_local = torch.stack(
        [math_utils.quat_apply(bin_quat_inv, rel[:, object_id, :]) for object_id in range(rel.shape[1])],
        dim=1,
    )

    half_x, half_y, max_z = bin_inner_half_extents
    object_half_extents = _object_half_extents(env, object_asset_names)
    footprint_radius = torch.linalg.vector_norm(object_half_extents[..., :2], dim=2)
    inside = (
        ((torch.abs(rel_local[..., 0]) + footprint_radius) <= half_x)
        & ((torch.abs(rel_local[..., 1]) + footprint_radius) <= half_y)
        & ((rel_local[..., 2] - object_half_extents[..., 2]) >= 0.012)
        & ((rel_local[..., 2] + object_half_extents[..., 2]) <= max_z)
    )
    all_active_inside = torch.all(torch.where(active, inside, torch.ones_like(inside)), dim=1)
    success_now = all_active_inside & _task_is(env, TASK_BIN)

    if not hasattr(env, "_so101_bin_success_counter"):
        env._so101_bin_success_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._so101_bin_success_counter = torch.where(
        success_now,
        env._so101_bin_success_counter + 1,
        torch.zeros_like(env._so101_bin_success_counter),
    )
    return env._so101_bin_success_counter >= _confirmation_steps(env, confirm_time_s, confirm_steps)


def next_to_success(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    success_distance: float = SPATIAL_SUCCESS_DISTANCE_M,
    confirm_steps: int | None = None,
    confirm_time_s: float = DEFAULT_SUCCESS_CONFIRM_TIME_S,
) -> torch.Tensor:
    """Success for ``Place object 1 next to object 2``."""

    positions = _object_positions(env, object_asset_names)
    half_extents = _object_half_extents(env, object_asset_names)
    target = _gather_by_index(positions, _target_indices(env))
    target_half = _gather_by_index(half_extents, _target_indices(env))
    ref_ids = _referent_indices(env)[:, 0]
    referent = _gather_by_index(positions, ref_ids)
    referent_half = _gather_by_index(half_extents, ref_ids)
    surface_distance = _aabb_surface_distance_xy(target, target_half, referent, referent_half)
    success_now = (surface_distance <= success_distance) & (~_target_is_on_top(env, object_asset_names))
    success_now &= _task_is(env, TASK_NEXT_TO)

    if not hasattr(env, "_so101_next_to_success_counter"):
        env._so101_next_to_success_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._so101_next_to_success_counter = torch.where(
        success_now,
        env._so101_next_to_success_counter + 1,
        torch.zeros_like(env._so101_next_to_success_counter),
    )
    return env._so101_next_to_success_counter >= _confirmation_steps(env, confirm_time_s, confirm_steps)


def between_success(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    centered_tolerance: float = BETWEEN_LINE_TOLERANCE_M,
    min_segment_fraction: float = 0.18,
    confirm_steps: int | None = None,
    confirm_time_s: float = DEFAULT_SUCCESS_CONFIRM_TIME_S,
) -> torch.Tensor:
    """Success for ``Place object 1 between object 2 and object 3``."""

    positions = _object_positions(env, object_asset_names)
    target = _gather_by_index(positions, _target_indices(env))[:, :2]
    refs = _referent_indices(env)
    ref_a = _gather_by_index(positions, refs[:, 0])[:, :2]
    ref_b = _gather_by_index(positions, refs[:, 1])[:, :2]

    segment = ref_b - ref_a
    segment_len_sq = torch.clamp(torch.sum(segment * segment, dim=1), min=1.0e-6)
    t = torch.sum((target - ref_a) * segment, dim=1) / segment_len_sq
    projection = ref_a + t.unsqueeze(1) * segment
    perpendicular = torch.linalg.vector_norm(target - projection, dim=1)
    centered = (t >= min_segment_fraction) & (t <= 1.0 - min_segment_fraction)
    success_now = centered & (perpendicular <= centered_tolerance) & (~_target_is_on_top(env, object_asset_names))
    success_now &= _task_is(env, TASK_BETWEEN)

    if not hasattr(env, "_so101_between_success_counter"):
        env._so101_between_success_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._so101_between_success_counter = torch.where(
        success_now,
        env._so101_between_success_counter + 1,
        torch.zeros_like(env._so101_between_success_counter),
    )
    return env._so101_between_success_counter >= _confirmation_steps(env, confirm_time_s, confirm_steps)


def _direction_vectors(device: str) -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 1.0],  # left
            [0.0, -1.0],  # right
            [1.0, 0.0],  # forward
            [-1.0, 0.0],  # backward
        ],
        dtype=torch.float32,
        device=device,
    )


def _direction_axis_and_sign(direction_id: int) -> tuple[int, float]:
    if direction_id == DIRECTIONS.index("left"):
        return (1, 1.0)
    if direction_id == DIRECTIONS.index("right"):
        return (1, -1.0)
    if direction_id == DIRECTIONS.index("forward"):
        return (0, 1.0)
    return (0, -1.0)


def _lane_intersects(
    target_pos: torch.Tensor,
    target_half: torch.Tensor,
    boundary_pos: torch.Tensor,
    boundary_half: torch.Tensor,
    lateral_axis: int,
) -> bool:
    center_distance = abs(float(target_pos[lateral_axis].item() - boundary_pos[lateral_axis].item()))
    lane_half_width = float(target_half[lateral_axis].item() + boundary_half[lateral_axis].item())
    return center_distance <= lane_half_width + SPATIAL_SUCCESS_DISTANCE_M


def _ensure_move_boundary_cache(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    table_bounds: dict[str, tuple[float, float]],
) -> None:
    """Pick the nearest directional boundary at reset for each move episode."""

    if hasattr(env, "_so101_move_boundary_coords") and hasattr(env, "_so101_move_boundary_ids"):
        return

    positions = getattr(env, "_so101_initial_object_pos_w", _object_positions(env, object_asset_names))
    half_extents = _object_half_extents(env, object_asset_names)
    active = _active_mask(env, object_asset_names)
    target_ids = _target_indices(env)
    direction_ids = _direction_indices(env)
    bin_positions = getattr(env, "_so101_initial_bin_pos_w", None)
    if bin_positions is None:
        bin_positions = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)
    bin_half_extents = _bin_half_extents(env)

    boundary_coords = torch.empty(env.num_envs, dtype=torch.float32, device=env.device)
    boundary_ids = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    for env_id in range(env.num_envs):
        axis, sign = _direction_axis_and_sign(int(direction_ids[env_id].item()))
        lateral_axis = 1 - axis
        target_id = int(target_ids[env_id].item())
        target_pos = positions[env_id, target_id]
        target_half = half_extents[env_id, target_id]
        target_front = float(target_pos[axis].item() + sign * target_half[axis].item())
        table_coord = table_bounds["x"][1 if sign > 0.0 else 0] if axis == 0 else table_bounds["y"][
            1 if sign > 0.0 else 0
        ]
        candidates: list[tuple[float, float, int]] = [(max(sign * (table_coord - target_front), 0.0), table_coord, -1)]

        for object_id in torch.nonzero(active[env_id], as_tuple=False).flatten().tolist():
            if object_id == target_id:
                continue
            boundary_pos = positions[env_id, object_id]
            boundary_half = half_extents[env_id, object_id]
            if not _lane_intersects(target_pos, target_half, boundary_pos, boundary_half, lateral_axis):
                continue
            surface_coord = float(boundary_pos[axis].item() - sign * boundary_half[axis].item())
            gap = sign * (surface_coord - target_front)
            if gap >= -0.002:
                candidates.append((max(gap, 0.0), surface_coord, object_id))

        if _lane_intersects(target_pos, target_half, bin_positions[env_id], bin_half_extents[env_id], lateral_axis):
            bin_surface_coord = float(bin_positions[env_id, axis].item() - sign * bin_half_extents[env_id, axis].item())
            bin_gap = sign * (bin_surface_coord - target_front)
            if bin_gap >= -0.002:
                candidates.append((max(bin_gap, 0.0), bin_surface_coord, -2))

        _gap, boundary_coord, boundary_id = min(candidates, key=lambda candidate: candidate[0])
        boundary_coords[env_id] = boundary_coord
        boundary_ids[env_id] = boundary_id

    env._so101_move_boundary_coords = boundary_coords
    env._so101_move_boundary_ids = boundary_ids


def _move_boundary_distance(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    table_bounds: dict[str, tuple[float, float]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _ensure_move_boundary_cache(env, object_asset_names, table_bounds)
    positions = _object_positions(env, object_asset_names)
    target = _gather_by_index(positions, _target_indices(env))[:, :2]
    initial = _gather_by_index(env._so101_initial_object_pos_w, _target_indices(env))[:, :2]
    desired = _direction_vectors(env.device)[_direction_indices(env)]
    delta = target - initial
    progress = torch.sum(delta * desired, dim=1)
    lateral = torch.linalg.vector_norm(delta - progress.unsqueeze(1) * desired, dim=1)

    along_x = torch.abs(desired[:, 0]) > 0.0
    axis_coord = torch.where(along_x, target[:, 0], target[:, 1])
    axis_sign = torch.where(along_x, desired[:, 0], desired[:, 1])
    boundary_distance = axis_sign * (env._so101_move_boundary_coords - axis_coord)
    return boundary_distance, progress, lateral, target


def move_success(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    table_bounds: dict[str, tuple[float, float]] | None = None,
    boundary_distance: float = SPATIAL_SUCCESS_DISTANCE_M,
    straightness_tolerance: float = SPATIAL_SUCCESS_DISTANCE_M,
    confirm_steps: int | None = None,
    confirm_time_s: float = DEFAULT_SUCCESS_CONFIRM_TIME_S,
) -> torch.Tensor:
    """Success for ``Move object direction`` against the nearest directional boundary."""

    if table_bounds is None:
        table_bounds = {"x": (0.08, 0.45), "y": (-0.20, 0.20)}

    distance_to_boundary, progress, lateral, _target = _move_boundary_distance(
        env, object_asset_names, table_bounds
    )
    close_to_boundary = (distance_to_boundary >= 0.0) & (distance_to_boundary <= boundary_distance)

    success_now = (
        (progress > 0.0)
        & close_to_boundary
        & (lateral <= straightness_tolerance)
        & (~_target_is_on_top(env, object_asset_names))
        & _task_is(env, TASK_MOVE)
    )

    if not hasattr(env, "_so101_move_success_counter"):
        env._so101_move_success_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._so101_move_success_counter = torch.where(
        success_now,
        env._so101_move_success_counter + 1,
        torch.zeros_like(env._so101_move_success_counter),
    )
    return env._so101_move_success_counter >= _confirmation_steps(env, confirm_time_s, confirm_steps)


def task_success(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    bin_name: str,
    table_bounds: dict[str, tuple[float, float]] | None = None,
    min_episode_time_s: float = 5.0,
    confirm_time_s: float = DEFAULT_SUCCESS_CONFIRM_TIME_S,
) -> torch.Tensor:
    """Dispatch to the success condition for the active benchmark family."""

    success = (
        bin_success(env, object_asset_names, bin_name, confirm_time_s=confirm_time_s)
        | next_to_success(env, object_asset_names, confirm_time_s=confirm_time_s)
        | between_success(env, object_asset_names, confirm_time_s=confirm_time_s)
        | move_success(env, object_asset_names, table_bounds, confirm_time_s=confirm_time_s)
    )
    return success & _episode_age_at_least(env, min_episode_time_s)


def _update_grasp_attempts(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    jaw_joint_name: str,
    jaw_close_delta: float,
    jaw_open_fraction: float,
    object_distance_threshold: float,
):
    """Count one target-object attempt for each armed SO-101 jaw close cycle."""

    robot = env.scene[robot_cfg.name]
    if not hasattr(env, "_so101_jaw_joint_id"):
        jaw_ids, _ = robot.find_joints(jaw_joint_name)
        env._so101_jaw_joint_id = jaw_ids[0]

    jaw_pos = robot.data.joint_pos[:, env._so101_jaw_joint_id]
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]

    if not hasattr(env, "_so101_grasp_attempt_counts"):
        env._so101_grasp_attempt_counts = torch.zeros(
            (env.num_envs, len(object_asset_names)), dtype=torch.long, device=env.device
        )
    if not hasattr(env, "_so101_grasp_armed"):
        env._so101_grasp_armed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if getattr(env, "_so101_grasp_arm_jaw_pos", None) is None:
        env._so101_grasp_arm_jaw_pos = jaw_pos.clone()

    jaw_limits = robot.data.joint_pos_limits[:, env._so101_jaw_joint_id]
    jaw_lower = torch.minimum(jaw_limits[:, 0], jaw_limits[:, 1])
    jaw_upper = torch.maximum(jaw_limits[:, 0], jaw_limits[:, 1])
    jaw_open_threshold = jaw_lower + jaw_open_fraction * (jaw_upper - jaw_lower)

    jaw_is_open = jaw_pos >= jaw_open_threshold
    was_armed = env._so101_grasp_armed
    newly_armed = jaw_is_open & (~was_armed)
    arm_jaw_pos = torch.where(newly_armed, jaw_pos, env._so101_grasp_arm_jaw_pos)
    arm_jaw_pos = torch.where(was_armed | newly_armed, torch.maximum(arm_jaw_pos, jaw_pos), arm_jaw_pos)

    close_cycle = (was_armed | newly_armed) & ((arm_jaw_pos - jaw_pos) >= jaw_close_delta)

    object_pos_w = _object_positions(env, object_asset_names)
    object_dist = torch.linalg.vector_norm(object_pos_w - ee_pos_w.unsqueeze(1), dim=2)
    eligible = _attempt_object_mask(env, object_asset_names)
    masked_dist = torch.where(eligible, object_dist, torch.full_like(object_dist, torch.inf))
    nearest_dist, nearest_object_ids = torch.min(masked_dist, dim=1)
    near_object = nearest_dist <= object_distance_threshold
    counted_attempts = close_cycle & near_object
    counted_env_ids = torch.nonzero(counted_attempts, as_tuple=False).flatten()
    if counted_env_ids.numel() > 0:
        env._so101_grasp_attempt_counts[counted_env_ids, nearest_object_ids[counted_env_ids]] += 1

    env._so101_grasp_armed = (was_armed | jaw_is_open) & (~close_cycle)
    env._so101_grasp_arm_jaw_pos = arm_jaw_pos


def _ensure_failure_displacement_baseline(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    bin_name: str,
    baseline_time_s: float,
) -> torch.Tensor:
    if not hasattr(env, "_so101_failure_object_pos_w"):
        env._so101_failure_object_pos_w = env._so101_initial_object_pos_w.clone()
    if not hasattr(env, "_so101_failure_bin_pos_w"):
        env._so101_failure_bin_pos_w = env._so101_initial_bin_pos_w.clone()
    if not hasattr(env, "_so101_failure_baseline_recorded"):
        env._so101_failure_baseline_recorded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    ready_to_record = _episode_age_at_least(env, baseline_time_s) & (~env._so101_failure_baseline_recorded)
    if torch.any(ready_to_record):
        mark_benchmark_robot_start(env, object_asset_names, bin_name, env_ids=ready_to_record)

    return env._so101_failure_baseline_recorded


def benchmark_failure(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    bin_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    jaw_joint_name: str = "Jaw",
    jaw_close_delta: float = 0.06,
    jaw_open_fraction: float = 0.5,
    grasp_attempt_object_distance: float = GRASP_ATTEMPT_OBJECT_DISTANCE_M,
    max_grasp_attempts: int = 3,
    bin_displacement_limit: float = BIN_DISPLACEMENT_LIMIT_M,
    non_target_displacement_limit: float = NON_TARGET_DISPLACEMENT_LIMIT_M,
    boundary_displacement_limit: float = BOUNDARY_DISPLACEMENT_LIMIT_M,
    min_episode_time_s: float = 5.0,
    displacement_baseline_time_s: float = 1.0,
    table_bounds: dict[str, tuple[float, float]] | None = None,
) -> torch.Tensor:
    """Cross-task failure conditions from the paper appendix.

    The term covers the measurable simulator-side rules: max grasp attempts,
    bin displacement, non-target object displacement, and moved move-boundaries.
    Qualitative labels such as semantic error and bad grasp strategy remain
    annotation categories rather than automatic simulator events.
    """

    if not hasattr(env, "_so101_initial_object_pos_w"):
        env._so101_failure_reasons = [FAILURE_REASON_NONE for _ in range(env.num_envs)]
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if table_bounds is None:
        table_bounds = {"x": (0.08, 0.45), "y": (-0.20, 0.20)}

    _update_grasp_attempts(
        env,
        object_asset_names=object_asset_names,
        robot_cfg=robot_cfg,
        ee_frame_cfg=ee_frame_cfg,
        jaw_joint_name=jaw_joint_name,
        jaw_close_delta=jaw_close_delta,
        jaw_open_fraction=jaw_open_fraction,
        object_distance_threshold=grasp_attempt_object_distance,
    )
    baseline_recorded = _ensure_failure_displacement_baseline(
        env,
        object_asset_names=object_asset_names,
        bin_name=bin_name,
        baseline_time_s=displacement_baseline_time_s,
    )

    active = _active_mask(env, object_asset_names)
    # The close that raises a target count to three is still a usable attempt.
    exhausted_attempts = env._so101_grasp_attempt_counts > max_grasp_attempts
    attempt_failure = torch.any(exhausted_attempts & _attempt_object_mask(env, object_asset_names), dim=1)

    bin_asset: RigidObject = env.scene[bin_name]
    bin_displacement = torch.linalg.vector_norm(bin_asset.data.root_pos_w - env._so101_failure_bin_pos_w, dim=1)
    bin_failure = bin_displacement > bin_displacement_limit

    object_pos_w = _object_positions(env, object_asset_names)
    object_displacement = torch.linalg.vector_norm(object_pos_w - env._so101_failure_object_pos_w, dim=2)
    target_ids = _target_indices(env)
    target_mask = torch.zeros_like(active)
    target_mask[torch.arange(env.num_envs, device=env.device), target_ids] = True
    instruction_task = ~_task_is(env, TASK_BIN)
    non_target_moved = torch.any((object_displacement > non_target_displacement_limit) & active & (~target_mask), dim=1)
    non_target_moved = non_target_moved & instruction_task

    _ensure_move_boundary_cache(env, object_asset_names, table_bounds)
    boundary_moved = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    boundary_object_ids = env._so101_move_boundary_ids
    object_boundary_env_ids = torch.nonzero(boundary_object_ids >= 0, as_tuple=False).flatten()
    if object_boundary_env_ids.numel() > 0:
        boundary_moved[object_boundary_env_ids] = (
            object_displacement[object_boundary_env_ids, boundary_object_ids[object_boundary_env_ids]]
            > boundary_displacement_limit
        )
    move_boundary_failure = boundary_moved & _task_is(env, TASK_MOVE)
    distance_to_boundary, _progress, _lateral, _target = _move_boundary_distance(
        env, object_asset_names, table_bounds
    )
    move_past_boundary = (distance_to_boundary < -0.002) & _task_is(env, TASK_MOVE)
    target_on_top = _target_is_on_top(env, object_asset_names) & (~_task_is(env, TASK_BIN))

    failure = (
        attempt_failure
        | bin_failure
        | non_target_moved
        | move_boundary_failure
        | move_past_boundary
        | target_on_top
    )
    aged_failure = failure & _episode_age_at_least(env, min_episode_time_s) & baseline_recorded

    env._so101_failure_reasons = [FAILURE_REASON_NONE for _ in range(env.num_envs)]
    for env_id in torch.nonzero(aged_failure, as_tuple=False).flatten().tolist():
        reasons = []
        if bool(attempt_failure[env_id].item()):
            reasons.append(FAILURE_REASON_MAX_GRASP_ATTEMPTS)
        if bool(bin_failure[env_id].item()):
            reasons.append(FAILURE_REASON_BIN_DISPLACED)
        if bool(non_target_moved[env_id].item()):
            reasons.append(FAILURE_REASON_NON_TARGET_MOVED)
        if bool(move_boundary_failure[env_id].item()):
            reasons.append(FAILURE_REASON_MOVE_BOUNDARY_MOVED)
        if bool(move_past_boundary[env_id].item()):
            reasons.append(FAILURE_REASON_MOVE_PAST_BOUNDARY)
        if bool(target_on_top[env_id].item()):
            reasons.append(FAILURE_REASON_TARGET_ON_TOP)
        env._so101_failure_reasons[env_id] = "+".join(reasons)

    return aged_failure
