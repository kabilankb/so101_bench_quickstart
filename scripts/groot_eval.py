# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Run SO-101 Bench with a remote GR00T policy server."""

from __future__ import annotations

import argparse
import json
import math
import random

from isaaclab.app import AppLauncher


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("1", "true", "t", "yes", "y", "on"):
        return True
    if value in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


parser = argparse.ArgumentParser(description="SO-101 Bench GR00T remote-policy evaluator.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="So101Bench-Bin-v0", help="Isaac Lab task name.")
parser.add_argument("--seed", type=int, default=1984, help="Environment seed.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate.")
parser.add_argument("--policy_host", type=str, default="localhost", help="GR00T policy server host.")
parser.add_argument("--policy_port", type=int, default=5555, help="GR00T policy server port.")
parser.add_argument("--action_horizon", type=int, default=16, help="Action steps to execute per server query.")
parser.add_argument(
    "--initial_hold_time_s",
    type=float,
    default=1.0,
    help="Seconds to hold the initial joint pose before recording overhead_init and querying GR00T.",
)
parser.add_argument(
    "--lang_instruction",
    type=str,
    default=None,
    help="Fixed language instruction. If omitted, the env-generated instruction for each reset is used.",
)
parser.add_argument(
    "--rename_map",
    type=str,
    default=None,
    help=(
        "JSON map from sim camera names to policy names. By default the sim wrist camera is sent as "
        '\'front\' to match the SO100/SO101 real-robot GR00T scripts. Example: '
        '\'{"wrist":"ego","overhead":"external"}\'.'
    ),
)
parser.add_argument(
    "--use_overhead_init",
    nargs="?",
    const=True,
    default=False,
    type=_str_to_bool,
    help=(
        "Send the settled overhead frame captured when robot control starts as video.overhead_init on every "
        "GR00T request. "
        "Accepts either '--use_overhead_init' or '--use_overhead_init true'."
    ),
)
parser.add_argument(
    "--overhead_init_key",
    type=str,
    default="overhead_init",
    help="Policy video key for the fixed settled overhead frame used by WM-conditioned checkpoints.",
)
parser.add_argument(
    "--overhead_init_camera",
    type=str,
    default="overhead",
    help="Sim camera name to capture for the fixed overhead-init frame.",
)
parser.add_argument(
    "--policy_image_width",
    type=int,
    default=640,
    help="Resize every policy video frame to this width before sending it to GR00T. Use 0 to disable resizing.",
)
parser.add_argument(
    "--policy_image_height",
    type=int,
    default=480,
    help="Resize every policy video frame to this height before sending it to GR00T. Use 0 to disable resizing.",
)
parser.add_argument(
    "--inspect_initial_scene",
    action="store_true",
    default=False,
    help="Reset the task, print/view the initial object poses, and exit only when the Isaac app closes without stepping physics.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import so101_bench.tasks  # noqa: F401
from so101_bench.mdp import mark_benchmark_robot_start
from so101_bench.tasks.direct.so101_bench.so101_bench_env_cfg import OBJECT_ASSET_NAMES
from so101_bench.utils.groot import GR00TRemotePolicy


def _discover_cameras(env) -> dict[str, dict[str, int]]:
    cameras = {}
    for scene_key in env.unwrapped.scene.keys():
        if not scene_key.startswith("camera_"):
            continue
        camera_cfg = getattr(env.unwrapped.scene.cfg, scene_key)
        camera_name = scene_key.replace("camera_", "")
        cameras[camera_name] = {"height": camera_cfg.height, "width": camera_cfg.width}
        print(f"[INFO]: Found camera '{camera_name}' ({camera_cfg.width}x{camera_cfg.height})")
    return cameras


def _instruction(env, override: str | None) -> str:
    if override:
        return override
    return getattr(env.unwrapped, "so101_bench_instruction", "Place each object in the plastic bin.")


def _rename_map(raw_map: str | None) -> dict[str, str]:
    rename_map = {"wrist": "front", "overhead": "overhead"}
    if raw_map:
        rename_map.update(json.loads(raw_map))
    return rename_map


def _episode_end_reason(env, terminated, truncated, term_log: dict) -> str:
    if bool(term_log.get("Episode_Termination/success", 0.0) > 0.0):
        return "success"

    failure_reasons = getattr(env.unwrapped, "_so101_failure_reasons", None)
    if failure_reasons:
        active_env_ids = torch.nonzero(terminated, as_tuple=False).flatten().tolist()
        for env_id in active_env_ids:
            reason = failure_reasons[env_id]
            if reason != "none":
                return reason

    if bool(term_log.get("Episode_Termination/failure", 0.0) > 0.0):
        return "failure"
    if bool(truncated.any().item()):
        return "time_out"
    return "unknown"


def _begin_robot_control(env, policy: GR00TRemotePolicy, obs: dict) -> None:
    mark_benchmark_robot_start(
        env.unwrapped,
        object_asset_names=OBJECT_ASSET_NAMES,
        bin_name="plastic_bin",
        force_robot_start_time=True,
    )
    policy.set_episode_initial_observation(obs["visual"])


def _print_initial_scene(env) -> None:
    unwrapped = env.unwrapped
    print(f"[INFO]: Episode instruction: {getattr(unwrapped, 'so101_bench_instruction', '')}")

    active_mask = getattr(unwrapped, "_so101_active_object_mask", None)
    for object_id, asset_name in enumerate(OBJECT_ASSET_NAMES):
        asset = unwrapped.scene[asset_name]
        pos = asset.data.root_pos_w[0].detach().cpu().tolist()
        active = bool(active_mask[0, object_id].item()) if active_mask is not None else True
        state = "active" if active else "inactive"
        print(
            f"[INFO]: Initial {asset_name} ({state}): "
            f"x={pos[0]:.5f}, y={pos[1]:.5f}, z={pos[2]:.5f}"
        )

    bin_asset = unwrapped.scene["plastic_bin"]
    bin_pos = bin_asset.data.root_pos_w[0].detach().cpu().tolist()
    print(f"[INFO]: Initial plastic_bin: x={bin_pos[0]:.5f}, y={bin_pos[1]:.5f}, z={bin_pos[2]:.5f}")


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed

    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)

    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    control_dt = float(env.unwrapped.step_dt)
    physics_dt = float(env.unwrapped.cfg.sim.dt)
    render_dt = physics_dt * int(env.unwrapped.cfg.sim.render_interval)
    initial_hold_steps = max(0, math.ceil(args_cli.initial_hold_time_s / control_dt))
    print(
        "[INFO]: Timing: "
        f"physics_dt={physics_dt:.6f}s, control_dt={control_dt:.6f}s, "
        f"render_dt={render_dt:.6f}s, action_chunk={args_cli.action_horizon * control_dt:.3f}s"
    )
    if initial_hold_steps > 0:
        print(f"[INFO]: Initial hold: {initial_hold_steps} steps ({initial_hold_steps * control_dt:.3f}s)")

    cameras = _discover_cameras(env)
    if not cameras:
        raise RuntimeError("No cameras were found. GR00T inference requires visual observations.")

    if args_cli.inspect_initial_scene:
        env.reset()
        _print_initial_scene(env)
        print("[INFO]: Inspecting initial scene. Close the Isaac app window to exit; physics is not being stepped.")
        while simulation_app.is_running():
            simulation_app.update()
        env.close()
        return

    rename_map = _rename_map(args_cli.rename_map)
    print(f"[INFO]: Policy camera map: {rename_map}")
    if args_cli.use_overhead_init:
        print(
            "[INFO]: WM overhead-init enabled: "
            f"rgb_{args_cli.overhead_init_camera} -> video.{args_cli.overhead_init_key}"
        )
    image_size = (
        (args_cli.policy_image_width, args_cli.policy_image_height)
        if args_cli.policy_image_width > 0 and args_cli.policy_image_height > 0
        else None
    )
    if image_size is not None:
        print(f"[INFO]: Resizing policy video frames to {image_size[0]}x{image_size[1]}")

    policy = GR00TRemotePolicy(
        device=env.unwrapped.device,
        cameras=cameras,
        host=args_cli.policy_host,
        port=args_cli.policy_port,
        action_horizon=args_cli.action_horizon,
        lang_instruction=args_cli.lang_instruction or "Place each object in the plastic bin.",
        rename_map=rename_map,
        use_overhead_init=args_cli.use_overhead_init,
        overhead_init_camera=args_cli.overhead_init_camera,
        overhead_init_key=args_cli.overhead_init_key,
        image_size=image_size,
    )
    policy.connect()

    obs, _ = env.reset()
    policy.set_language_instruction(_instruction(env, args_cli.lang_instruction))
    policy.reset()
    print(f"[INFO]: Episode instruction: {policy.lang_instruction}")

    initial_action = torch.tensor(
        [-0.252462, -1.471487, 0.785040, 1.283341, -0.094108, -0.150151],
        device=env.unwrapped.device,
    )
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    step = 0
    episodes = 0
    successes = 0
    robot_control_started = False

    while simulation_app.is_running():
        with torch.inference_mode():
            if step < initial_hold_steps:
                actions[:] = initial_action
            else:
                if not robot_control_started:
                    _begin_robot_control(env, policy, obs)
                    robot_control_started = True
                joint_positions = obs["policy"]["joint_pos_obs"][0].clone()
                actions[:] = policy.get_action(joint_positions, obs["visual"])

            obs, _rewards, terminated, truncated, info = env.step(actions)
            step += 1

            is_done = bool(terminated.any().item() or truncated.any().item())
            if not is_done:
                continue

            term_log = info.get("log", {})
            is_success = bool(term_log.get("Episode_Termination/success", 0.0) > 0.0)
            end_reason = _episode_end_reason(env, terminated, truncated, term_log)
            episodes += 1
            successes += int(is_success)
            print(f"[INFO]: Episode {episodes}/{args_cli.num_episodes}: success={is_success}, reason={end_reason}")

            if episodes >= args_cli.num_episodes:
                rate = 100.0 * successes / max(episodes, 1)
                print(f"[INFO]: Success Rate: {successes}/{episodes} ({rate:.1f}%)")
                break

            obs, _ = env.reset()
            policy.set_language_instruction(_instruction(env, args_cli.lang_instruction))
            policy.reset()
            print(f"[INFO]: Episode instruction: {policy.lang_instruction}")
            step = 0
            robot_control_started = False

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
