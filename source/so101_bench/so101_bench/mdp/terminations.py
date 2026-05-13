"""Success and failure termination terms for SO-101 Bench."""

from __future__ import annotations

import math

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from so101_bench.benchmark import (
    BIN_DISPLACEMENT_LIMIT_M,
    BOUNDARY_DISPLACEMENT_LIMIT_M,
    DIRECTIONS,
    GRASP_ATTEMPT_EE_MOTION_M,
    NON_TARGET_DISPLACEMENT_LIMIT_M,
    SPATIAL_SUCCESS_DISTANCE_M,
    TASK_BETWEEN,
    TASK_BIN,
    TASK_MOVE,
    TASK_NEXT_TO,
)

from .resets import mark_benchmark_robot_start

FAILURE_REASON_NONE = "none"
FAILURE_REASON_MAX_GRASP_ATTEMPTS = "max_grasp_attempts"
FAILURE_REASON_BIN_DISPLACED = "bin_displaced"
FAILURE_REASON_NON_TARGET_MOVED = "non_target_moved"
FAILURE_REASON_MOVE_BOUNDARY_MOVED = "move_boundary_moved"

DEFAULT_SUCCESS_CONFIRM_TIME_S = 0.25


def _active_mask(env: ManagerBasedRLEnv, object_asset_names: list[str]) -> torch.Tensor:
    if hasattr(env, "_so101_active_object_mask"):
        return env._so101_active_object_mask
    return torch.ones((env.num_envs, len(object_asset_names)), dtype=torch.bool, device=env.device)


def _object_positions(env: ManagerBasedRLEnv, object_asset_names: list[str]) -> torch.Tensor:
    return torch.stack([env.scene[name].data.root_pos_w for name in object_asset_names], dim=1)


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
    inside = (
        (torch.abs(rel_local[..., 0]) <= half_x)
        & (torch.abs(rel_local[..., 1]) <= half_y)
        & (rel_local[..., 2] >= 0.012)
        & (rel_local[..., 2] <= max_z)
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
    target = _gather_by_index(positions, _target_indices(env))
    ref_ids = _referent_indices(env)[:, 0]
    referent = _gather_by_index(positions, ref_ids)
    dist_xy = torch.linalg.vector_norm(target[:, :2] - referent[:, :2], dim=1)
    success_now = (dist_xy <= success_distance) & _task_is(env, TASK_NEXT_TO)

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
    centered_tolerance: float = SPATIAL_SUCCESS_DISTANCE_M,
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
    success_now = centered & (perpendicular <= centered_tolerance) & _task_is(env, TASK_BETWEEN)

    if not hasattr(env, "_so101_between_success_counter"):
        env._so101_between_success_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._so101_between_success_counter = torch.where(
        success_now,
        env._so101_between_success_counter + 1,
        torch.zeros_like(env._so101_between_success_counter),
    )
    return env._so101_between_success_counter >= _confirmation_steps(env, confirm_time_s, confirm_steps)


def move_success(
    env: ManagerBasedRLEnv,
    object_asset_names: list[str],
    table_bounds: dict[str, tuple[float, float]] | None = None,
    boundary_distance: float = SPATIAL_SUCCESS_DISTANCE_M,
    straightness_tolerance: float = SPATIAL_SUCCESS_DISTANCE_M,
    confirm_steps: int | None = None,
    confirm_time_s: float = DEFAULT_SUCCESS_CONFIRM_TIME_S,
) -> torch.Tensor:
    """Success for ``Move object direction`` with the tabletop edge as the boundary."""

    if table_bounds is None:
        table_bounds = {"x": (0.08, 0.45), "y": (-0.20, 0.20)}

    positions = _object_positions(env, object_asset_names)
    target = _gather_by_index(positions, _target_indices(env))[:, :2]
    initial = _gather_by_index(env._so101_initial_object_pos_w, _target_indices(env))[:, :2]
    delta = target - initial
    direction_ids = _direction_indices(env)

    direction_vecs = torch.tensor(
        [
            [0.0, 1.0],   # left
            [0.0, -1.0],  # right
            [1.0, 0.0],   # forward
            [-1.0, 0.0],  # backward
        ],
        dtype=torch.float32,
        device=env.device,
    )
    desired = direction_vecs[direction_ids]
    progress = torch.sum(delta * desired, dim=1)
    lateral = torch.linalg.vector_norm(delta - progress.unsqueeze(1) * desired, dim=1)

    x_min, x_max = table_bounds["x"]
    y_min, y_max = table_bounds["y"]
    boundary = torch.empty(env.num_envs, dtype=torch.float32, device=env.device)
    boundary[direction_ids == DIRECTIONS.index("left")] = y_max
    boundary[direction_ids == DIRECTIONS.index("right")] = y_min
    boundary[direction_ids == DIRECTIONS.index("forward")] = x_max
    boundary[direction_ids == DIRECTIONS.index("backward")] = x_min
    coord = torch.where(torch.abs(desired[:, 0]) > 0.0, target[:, 0], target[:, 1])
    close_to_boundary = torch.abs(boundary - coord) <= boundary_distance

    success_now = (
        (progress > 0.0)
        & close_to_boundary
        & (lateral <= straightness_tolerance)
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
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    jaw_joint_name: str,
    jaw_close_delta: float,
    ee_motion_threshold: float,
):
    robot = env.scene[robot_cfg.name]
    if not hasattr(env, "_so101_jaw_joint_id"):
        jaw_ids, _ = robot.find_joints(jaw_joint_name)
        env._so101_jaw_joint_id = jaw_ids[0]

    jaw_pos = robot.data.joint_pos[:, env._so101_jaw_joint_id]
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]

    if env._so101_prev_jaw_pos is None or env._so101_prev_ee_pos_w is None:
        env._so101_prev_jaw_pos = jaw_pos.clone()
        env._so101_prev_ee_pos_w = ee_pos_w.clone()
        return

    step = getattr(env, "common_step_counter", int(env.episode_length_buf.max().item()))
    current_step = torch.full((env.num_envs,), int(step), dtype=torch.long, device=env.device)
    already_updated = env._so101_last_attempt_step == current_step

    jaw_delta = jaw_pos - env._so101_prev_jaw_pos
    ee_motion = torch.linalg.vector_norm(ee_pos_w - env._so101_prev_ee_pos_w, dim=1)
    closed_after_motion = (jaw_delta < -jaw_close_delta) & (ee_motion > ee_motion_threshold) & (~already_updated)
    env._so101_grasp_attempt_count += closed_after_motion.long()
    env._so101_last_attempt_step = torch.where(closed_after_motion, current_step, env._so101_last_attempt_step)
    env._so101_prev_jaw_pos = jaw_pos.clone()
    env._so101_prev_ee_pos_w = ee_pos_w.clone()


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
    max_grasp_attempts: int = 3,
    bin_displacement_limit: float = BIN_DISPLACEMENT_LIMIT_M,
    non_target_displacement_limit: float = NON_TARGET_DISPLACEMENT_LIMIT_M,
    boundary_displacement_limit: float = BOUNDARY_DISPLACEMENT_LIMIT_M,
    min_episode_time_s: float = 5.0,
    displacement_baseline_time_s: float = 1.0,
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

    _update_grasp_attempts(
        env,
        robot_cfg=robot_cfg,
        ee_frame_cfg=ee_frame_cfg,
        jaw_joint_name=jaw_joint_name,
        jaw_close_delta=jaw_close_delta,
        ee_motion_threshold=GRASP_ATTEMPT_EE_MOTION_M,
    )
    baseline_recorded = _ensure_failure_displacement_baseline(
        env,
        object_asset_names=object_asset_names,
        bin_name=bin_name,
        baseline_time_s=displacement_baseline_time_s,
    )

    active = _active_mask(env, object_asset_names)
    bin_allowed_attempts = max_grasp_attempts * torch.clamp(active.sum(dim=1), min=1)
    single_target_allowed_attempts = torch.full_like(bin_allowed_attempts, max_grasp_attempts)
    allowed_attempts = torch.where(_task_is(env, TASK_BIN), bin_allowed_attempts, single_target_allowed_attempts)
    attempt_failure = env._so101_grasp_attempt_count >= allowed_attempts

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

    refs = _referent_indices(env)
    boundary_mask = torch.zeros_like(active)
    boundary_mask[torch.arange(env.num_envs, device=env.device), refs[:, 0]] = True
    boundary_mask[torch.arange(env.num_envs, device=env.device), refs[:, 1]] = True
    boundary_moved = torch.any((object_displacement > boundary_displacement_limit) & boundary_mask, dim=1)
    move_boundary_failure = boundary_moved & _task_is(env, TASK_MOVE)

    failure = attempt_failure | bin_failure | non_target_moved | move_boundary_failure
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
        env._so101_failure_reasons[env_id] = "+".join(reasons)

    return aged_failure
