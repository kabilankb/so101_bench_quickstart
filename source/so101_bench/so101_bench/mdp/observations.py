"""Observation terms for the SO-101 benchmark environments."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


def ee_frame_state(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the end-effector frame pose in the robot root frame."""

    robot = env.scene[robot_cfg.name]
    robot_root_pos, robot_root_quat = robot.data.root_pos_w, robot.data.root_quat_w
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]
    ee_frame_pos_robot, ee_frame_quat_robot = math_utils.subtract_frame_transforms(
        robot_root_pos, robot_root_quat, ee_frame_pos, ee_frame_quat
    )
    return torch.cat([ee_frame_pos_robot, ee_frame_quat_robot], dim=1)


def image_raw(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera_overhead"),
    data_type: str = "rgb",
) -> torch.Tensor:
    """Return a camera tensor without normalization or reshaping."""

    sensor = env.scene[sensor_cfg.name]
    return sensor.data.output[data_type].clone()

