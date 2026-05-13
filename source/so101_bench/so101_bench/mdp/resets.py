"""Reset events for SO-101 Bench."""

from __future__ import annotations

import math
import random

import torch
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import get_current_stage

from so101_bench.benchmark import (
    DIRECTIONS,
    MAX_GRASP_ATTEMPTS,
    TASK_BETWEEN,
    TASK_BIN,
    TASK_FAMILIES,
    TASK_MIXED,
    TASK_MOVE,
    TASK_NEXT_TO,
    task_instruction,
)

ROBOT_COLORS = {
    "orange": (0.876, 0.317, 0.132),
    "teal": (0.0, 0.8, 0.502),
    "white": (0.95, 0.95, 0.95),
    "black": (0.08, 0.08, 0.08),
}


def _env_step_dt(env) -> float:
    step_dt = getattr(env, "step_dt", env.cfg.sim.dt * env.cfg.decimation)
    return float(step_dt)


def _episode_age_s(env) -> torch.Tensor:
    return env.episode_length_buf.to(dtype=torch.float32) * _env_step_dt(env)


def _env_ids_tensor(env, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)

    env_ids = env_ids.to(device=env.device)
    if env_ids.dtype == torch.bool:
        return torch.nonzero(env_ids, as_tuple=False).flatten()
    return env_ids.to(dtype=torch.long)


def _object_positions(env, object_asset_names: list[str]) -> torch.Tensor:
    return torch.stack([env.scene[name].data.root_pos_w for name in object_asset_names], dim=1)


def _ensure_robot_start_buffers(env) -> None:
    if not hasattr(env, "_so101_robot_started_moving"):
        env._so101_robot_started_moving = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, "_so101_robot_start_step"):
        env._so101_robot_start_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    if not hasattr(env, "_so101_robot_start_time_s"):
        env._so101_robot_start_time_s = torch.full(
            (env.num_envs,), float("nan"), dtype=torch.float32, device=env.device
        )


def mark_benchmark_robot_start(
    env,
    object_asset_names: list[str],
    bin_name: str,
    env_ids: torch.Tensor | None = None,
    force_robot_start_time: bool = False,
) -> None:
    """Record settled displacement baselines and stamp when robot control starts."""

    if not hasattr(env, "_so101_initial_object_pos_w"):
        return

    env_ids = _env_ids_tensor(env, env_ids)
    if env_ids.numel() == 0:
        return

    if not hasattr(env, "_so101_failure_object_pos_w"):
        env._so101_failure_object_pos_w = env._so101_initial_object_pos_w.clone()
    if not hasattr(env, "_so101_failure_bin_pos_w"):
        env._so101_failure_bin_pos_w = env._so101_initial_bin_pos_w.clone()
    if not hasattr(env, "_so101_failure_baseline_recorded"):
        env._so101_failure_baseline_recorded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    baseline_env_ids = env_ids[~env._so101_failure_baseline_recorded[env_ids]]
    if baseline_env_ids.numel() > 0:
        env._so101_failure_object_pos_w[baseline_env_ids] = _object_positions(env, object_asset_names)[
            baseline_env_ids
        ]
        env._so101_failure_bin_pos_w[baseline_env_ids] = env.scene[bin_name].data.root_pos_w[baseline_env_ids]
        env._so101_failure_baseline_recorded[baseline_env_ids] = True

    _ensure_robot_start_buffers(env)
    if force_robot_start_time:
        start_env_ids = env_ids
    else:
        start_env_ids = env_ids[~env._so101_robot_started_moving[env_ids]]
    if start_env_ids.numel() == 0:
        return

    env._so101_robot_started_moving[start_env_ids] = True
    env._so101_robot_start_step[start_env_ids] = env.episode_length_buf[start_env_ids]
    env._so101_robot_start_time_s[start_env_ids] = _episode_age_s(env)[start_env_ids]


def randomize_robot_color(
    env,
    env_ids: torch.Tensor | None,
    color_names: list[str] | None = None,
):
    """Set the robot visual material to one of the supported SO-101 colors."""

    if color_names is None:
        color_names = list(ROBOT_COLORS.keys())
    color = ROBOT_COLORS[random.choice(color_names)]

    with Sdf.ChangeBlock():
        robot = env.scene["robot"]
        material_prim_path = robot.cfg.prim_path + "/Looks/material_a_3d_printed/Shader"
        material_prims = sim_utils.find_matching_prims(material_prim_path)
        if not material_prims:
            return
        material_prims[0].GetAttribute("inputs:diffuse_color_constant").Set(color)


def _sample_task_family(task_family: str) -> str:
    if task_family == TASK_MIXED:
        return random.choice(TASK_FAMILIES)
    return task_family


def _required_object_count(task_family: str, object_count_range: tuple[int, int]) -> int:
    low, high = object_count_range
    if task_family == TASK_NEXT_TO:
        low = max(low, 2)
    elif task_family == TASK_BETWEEN:
        low = max(low, 3)
    high = max(high, low)
    return random.randint(low, high)


def _sample_positions(
    count: int,
    table_bounds: dict[str, tuple[float, float]],
    min_spacing: float,
) -> list[tuple[float, float]]:
    positions: list[tuple[float, float]] = []
    x_min, x_max = table_bounds["x"]
    y_min, y_max = table_bounds["y"]

    for _ in range(count):
        candidate = (0.0, 0.0)
        for _attempt in range(100):
            candidate = (random.uniform(x_min, x_max), random.uniform(y_min, y_max))
            if all(math.dist(candidate, existing) >= min_spacing for existing in positions):
                break
        positions.append(candidate)
    return positions


def _fixed_positions(
    count: int,
    object_fixed_poses: tuple[tuple[float, float, float], ...],
) -> list[tuple[float, float]]:
    return [(pose[0], pose[1]) for pose in object_fixed_poses[:count]]


def _yaw_quat(yaw: float, device: str) -> torch.Tensor:
    return math_utils.quat_from_euler_xyz(
        torch.zeros(1, device=device),
        torch.zeros(1, device=device),
        torch.tensor([yaw], device=device),
    )[0]


def _rpy_quat(rpy: tuple[float, float, float], device: str) -> torch.Tensor:
    return math_utils.quat_from_euler_xyz(
        torch.tensor([rpy[0]], device=device),
        torch.tensor([rpy[1]], device=device),
        torch.tensor([rpy[2]], device=device),
    )[0]


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind()
    w2, x2, y2, z2 = q2.unbind()
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    )


def _bin_quat(yaw: float, root_rotation: tuple[float, float, float], device: str) -> torch.Tensor:
    return _quat_mul(_yaw_quat(yaw, device), _rpy_quat(root_rotation, device))


def _write_pose(
    env,
    asset: RigidObject | Articulation,
    env_id: int,
    pos: tuple[float, float, float],
    quat: torch.Tensor,
):
    env_ids = torch.tensor([env_id], dtype=torch.long, device=asset.device)
    root_pose = torch.zeros((1, 7), device=asset.device)
    root_pose[:, :3] = torch.tensor(pos, dtype=torch.float32, device=asset.device)
    root_pose[:, :3] += env.scene.env_origins[env_ids]
    root_pose[:, 3:7] = quat.to(asset.device).unsqueeze(0)
    asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros((1, 6), device=asset.device), env_ids=env_ids)
    return root_pose[0, :3]


def reset_benchmark_scene(
    env,
    env_ids: torch.Tensor,
    object_asset_names: list[str],
    bin_name: str,
    object_labels: list[str],
    task_family: str = TASK_MIXED,
    object_count_range: tuple[int, int] = (1, 4),
    table_bounds: dict[str, tuple[float, float]] | None = None,
    table_top_z: float = 0.032,
    min_object_spacing: float = 0.105,
    bin_fixed_pose: tuple[float, float, float] = (0.34, 0.16, 0.0),
    bin_root_rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    bin_z: float | None = None,
    object_fixed_poses: tuple[tuple[float, float, float], ...] | None = None,
    randomize_bin_for_bin_task: bool = True,
    bin_x_range: tuple[float, float] = (0.26, 0.43),
    bin_y_range: tuple[float, float] = (-0.19, 0.19),
    inactive_z: float = -0.35,
):
    """Randomize the benchmark task, plastic bin, and 1-4 tabletop objects.

    The function stores per-episode metadata on the environment under
    ``so101_bench_instruction`` and private tensor buffers consumed by the
    success/failure termination terms.
    """

    if table_bounds is None:
        table_bounds = {"x": (0.08, 0.45), "y": (-0.20, 0.20)}
    if bin_z is None:
        bin_z = table_top_z

    num_envs = env.num_envs
    num_objects = len(object_asset_names)
    device = env.device

    env._so101_task_family = [TASK_BIN for _ in range(num_envs)]
    env._so101_instruction_text = ["" for _ in range(num_envs)]
    env._so101_active_object_mask = torch.zeros((num_envs, num_objects), dtype=torch.bool, device=device)
    env._so101_target_object_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
    env._so101_referent_object_ids = torch.zeros((num_envs, 2), dtype=torch.long, device=device)
    env._so101_direction_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
    env._so101_initial_object_pos_w = torch.zeros((num_envs, num_objects, 3), dtype=torch.float32, device=device)
    env._so101_initial_bin_pos_w = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    env._so101_failure_object_pos_w = torch.zeros((num_envs, num_objects, 3), dtype=torch.float32, device=device)
    env._so101_failure_bin_pos_w = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    env._so101_failure_baseline_recorded = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._so101_robot_started_moving = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._so101_robot_start_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    env._so101_robot_start_time_s = torch.full((num_envs,), float("nan"), dtype=torch.float32, device=device)
    env._so101_grasp_attempt_count = torch.zeros(num_envs, dtype=torch.long, device=device)
    env._so101_max_grasp_attempts = MAX_GRASP_ATTEMPTS
    env._so101_prev_jaw_pos = None
    env._so101_prev_ee_pos_w = None
    env._so101_last_attempt_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    env._so101_bin_success_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
    env._so101_next_to_success_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
    env._so101_between_success_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
    env._so101_move_success_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
    env.so101_bench_episodes = []

    bin_asset: RigidObject = env.scene[bin_name]
    object_assets: list[RigidObject] = [env.scene[name] for name in object_asset_names]

    for env_id_tensor in env_ids:
        env_id = int(env_id_tensor.item())
        selected_task = _sample_task_family(task_family)
        active_count = _required_object_count(selected_task, object_count_range)
        active_count = min(active_count, num_objects)
        if object_fixed_poses is not None:
            if len(object_fixed_poses) < active_count:
                raise ValueError(
                    f"Need at least {active_count} fixed object poses, got {len(object_fixed_poses)}."
                )
            sampled_positions = _fixed_positions(active_count, object_fixed_poses)
        else:
            sampled_positions = _sample_positions(active_count, table_bounds, min_object_spacing)

        label_perm = list(object_labels)
        random.shuffle(label_perm)
        active_labels = label_perm[:active_count]
        direction = random.choice(DIRECTIONS)

        env._so101_task_family[env_id] = selected_task
        env._so101_active_object_mask[env_id, :active_count] = True
        env._so101_target_object_ids[env_id] = 0
        env._so101_referent_object_ids[env_id, 0] = 1 if active_count > 1 else 0
        env._so101_referent_object_ids[env_id, 1] = 2 if active_count > 2 else 1
        env._so101_direction_ids[env_id] = DIRECTIONS.index(direction)

        if selected_task == TASK_BIN and randomize_bin_for_bin_task:
            bin_x = random.uniform(*bin_x_range)
            bin_y = random.uniform(*bin_y_range)
            bin_yaw = random.uniform(-0.45, 0.45)
        else:
            bin_x, bin_y, bin_yaw = bin_fixed_pose

        env._so101_initial_bin_pos_w[env_id] = _write_pose(
            env,
            bin_asset,
            env_id,
            (bin_x, bin_y, bin_z),
            _bin_quat(bin_yaw, bin_root_rotation, device),
        )
        env._so101_failure_bin_pos_w[env_id] = env._so101_initial_bin_pos_w[env_id]

        for object_id, asset in enumerate(object_assets):
            default_z = float(asset.data.default_root_state[env_id, 2].item())
            if object_id < active_count:
                x, y = sampled_positions[object_id]
                z = default_z if default_z > table_top_z else table_top_z + 0.025
                yaw = (
                    object_fixed_poses[object_id][2]
                    if object_fixed_poses is not None
                    else random.uniform(-math.pi, math.pi)
                )
            else:
                x = table_bounds["x"][0] - 0.35
                y = table_bounds["y"][0] - 0.35 - 0.04 * object_id
                z = inactive_z
                yaw = 0.0

            env._so101_initial_object_pos_w[env_id, object_id] = _write_pose(
                env,
                asset,
                env_id,
                (x, y, z),
                _yaw_quat(yaw, device),
            )
            env._so101_failure_object_pos_w[env_id, object_id] = env._so101_initial_object_pos_w[
                env_id, object_id
            ]

        instruction = task_instruction(selected_task, active_labels, direction)
        env._so101_instruction_text[env_id] = instruction
        env.so101_bench_episodes.append(
            {
                "env_id": env_id,
                "task_family": selected_task,
                "instruction": instruction,
                "active_object_count": active_count,
                "active_labels": active_labels,
                "direction": direction if selected_task == TASK_MOVE else None,
            }
        )

    env.so101_bench_instruction = env._so101_instruction_text[int(env_ids[0].item())]
