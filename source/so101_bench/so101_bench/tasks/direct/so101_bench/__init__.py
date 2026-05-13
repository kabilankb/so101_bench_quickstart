"""Gym registrations for SO-101 Bench environments."""

import gymnasium as gym

_ENTRY_POINT = "isaaclab.envs:ManagerBasedRLEnv"
_CFG_MODULE = f"{__name__}.so101_bench_env_cfg"


def _register(task_id: str, cfg_name: str):
    gym.register(
        id=task_id,
        entry_point=_ENTRY_POINT,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_CFG_MODULE}:{cfg_name}",
        },
    )


_register("So101Bench-Mixed-v0", "So101BenchEnvCfg")
_register("So101Bench-Bin-v0", "So101BenchBinEnvCfg")
_register("So101Bench-NextTo-v0", "So101BenchNextToEnvCfg")
_register("So101Bench-Between-v0", "So101BenchBetweenEnvCfg")
_register("So101Bench-Move-v0", "So101BenchMoveEnvCfg")

