# SO-101 Bench Isaac Lab Environment

This extension turns the Isaac Lab template into an SO-101 Bench environment for
language-conditioned tabletop manipulation with a GR00T-N1.6 fine-tune. It uses
an orange SO-101 arm, a constant plastic bin, one to four tabletop objects, wrist
and overhead RGB-D cameras, and the four benchmark task families from the paper:

- `So101Bench-Bin-v0`: place each object, or the single object, in the plastic bin.
- `So101Bench-Bin-SingleObject-v0`: bin task with exactly one randomly selected object slot active on the table.
- `So101Bench-Bin-Object1-v0` through `So101Bench-Bin-Object4-v0`: bin task with a specific object slot active.
- `So101Bench-NextTo-v0`: place one object next to another.
- `So101Bench-Between-v0`: place one object between two referents.
- `So101Bench-Move-v0`: move one object in a commanded direction.
- `So101Bench-Mixed-v0`: sample among all four families.

The environment uses local USD assets for the bedroom tabletop, plastic bin, and
tabletop objects.

## Required Local Asset Path

The bundled scan at
`source/so101_bench/so101_bench/assets/usd/room_scan.usdc` is used by default
when present. To use a different bedroom/tabletop USD, update
`BEDROOM_TABLETOP_USD` in
`source/so101_bench/so101_bench/tasks/direct/so101_bench/so101_bench_env_cfg.py`.

For the bundled scan, `collision_mesh/Plane_002` is treated as the collidable
tabletop mesh, while `table_visual` and `room` remain visual-only. The scan is
authored at real-world scale, with the tabletop origin centered on the tabletop
surface, so the environment loads it with identity scale and rotation.

## Install

Use the Python interpreter that has Isaac Lab installed:

```bash
python -m pip install -e source/so101_bench
```

If Isaac Lab is not on your shell `python`, use your Isaac Lab launcher:

```bash
/home/truman/IsaacLab/isaaclab.sh -p -m pip install -e source/so101_bench
```

## Smoke Tests

List registered tasks:

```bash
/home/truman/IsaacLab/isaaclab.sh -p scripts/list_envs.py
```

Run the environment with a zero-action debug agent:

```bash
/home/truman/IsaacLab/isaaclab.sh -p scripts/zero_agent.py --task So101Bench-Bin-v0 --enable_cameras
```

Inspect the first JSONL episode without stepping physics:

```bash
/home/truman/IsaacLab/isaaclab.sh -p scripts/groot_eval.py \
  --task So101Bench-Bin-v0 \
  --episodes_jsonl tasks/real_gr00t_seen_1obj_WM.jsonl \
  --inspect_initial_scene
```

## GR00T Remote Inference

Start your GR00T-N1.6 fine-tuned policy server separately, then run:

```bash
/home/truman/IsaacLab/isaaclab.sh -p scripts/groot_eval.py \
  --task So101Bench-Bin-v0 \
  --episodes_jsonl tasks/real_gr00t_seen_1obj_WM.jsonl \
  --policy_host localhost \
  --policy_port 5555
```

`--episodes_jsonl` is required by `scripts/groot_eval.py`. It drives the
episode objects and instruction from each JSONL row. Every object name is
validated against `OBJECT_SPLITS` in
`so101_bench/benchmark.py` before evaluation. Object labels map to local USD
filenames by replacing spaces with underscores, so `"green shoes"` selects
`assets/usd/objects/green_shoes.usdc`. Slots whose object USDs contain multiple
rigid bodies are spawned as `AssetBaseCfg`; single-rigid-body objects use
`RigidObjectCfg`.

By default the evaluator runs every validated JSONL row. Use
`--num_episodes 10` to evaluate only the first 10 rows.

Rows must provide `objects` and `instruction`. The instruction must be one of
the four benchmark forms:

```json
{"objects": ["grey wires"], "instruction": "Place each object in the plastic bin"}
{"objects": ["black glasses", "silver glasses", "yellow toy car", "cardboard box"], "instruction": "Place the yellow toy car next to the silver glasses."}
{"objects": ["black glasses", "silver glasses", "yellow toy car", "cardboard box"], "instruction": "Place the cardboard box between the black glasses and the yellow toy car."}
{"objects": ["black glasses", "silver glasses", "yellow toy car", "cardboard box"], "instruction": "Move the cardboard box forwards."}
```

For WM-conditioned checkpoints that expect the real-robot `overhead_init`
camera stream, enable the fixed settled-frame input. The evaluator holds the
initial robot pose for 1 second by default, then records `overhead_init` and
starts querying GR00T:

```bash
/home/truman/IsaacLab/isaaclab.sh -p scripts/groot_eval.py \
  --task So101Bench-Bin-v0 \
  --episodes_jsonl tasks/real_gr00t_seen_1obj_WM.jsonl \
  --policy_host localhost \
  --policy_port 5555 \
  --action_horizon 16 \
  --use_overhead_init true
```

By default, the script sends the current JSONL row instruction to the policy.
To override only the policy language, pass a matching fixed instruction:

```bash
/home/truman/IsaacLab/isaaclab.sh -p scripts/groot_eval.py \
  --task So101Bench-Bin-v0 \
  --episodes_jsonl tasks/real_gr00t_seen_1obj_WM.jsonl \
  --lang_instruction "Place each object in the plastic bin."
```

The sim camera names are `wrist` and `overhead`. The evaluator sends the wrist
camera as policy key `front` by default to match the SO100/SO101 real-robot
GR00T scripts. If your fine-tune expects different camera keys, pass a rename
map:

```bash
--rename_map '{"wrist":"ego","overhead":"external"}'
```

The wrist camera defaults to a 640x480 render so the sim sends the same image
shape as the real-robot OpenCV command. To compare/tune policy frames:

```bash
/home/truman/IsaacLab/isaaclab.sh -p scripts/view_wrist_camera.py \
  --task So101Bench-Bin-v0 \
  --camera wrist \
  --steps 1 \
  --save_every 1
```

Useful wrist-camera tuning knobs live in
`source/so101_bench/so101_bench/tasks/direct/so101_bench/so101_bench_env_cfg.py`:

```python
INNOMAKER_WRIST_CAMERA_HORIZONTAL_FOV_DEG = 102.0
INNOMAKER_WRIST_CAMERA_POS = (-0.005, 0.060, -0.062)
INNOMAKER_WRIST_CAMERA_RPY_DEG = (-45.0, 0.0, 0.0)
```

## Paper-Derived Evaluation Logic

The simulator includes automatic checks for the measurable rules in the paper:

- maximum of three grasp attempts per target object;
- plastic bin displaced by more than 1 inch;
- non-target object displaced by more than 0.5 inches;
- move-task boundary object displaced by more than 0.5 inches;
- all active-object containment for bin placement;
- closest-surface and no-on-top checks for next-to placement;
- the between-task 1.5-inch COM-to-line rule plus centering and no-on-top checks;
- directional move boundaries selected from the nearest table edge, bin, or
  blocking object, with straightness, no-crossing, and no-on-top checks.

Qualitative appendix labels such as semantic error, bad grasp strategy,
occlusion-induced grasp failure, and failed reorientation remain annotation
categories. They are preserved in `so101_bench/benchmark.py` but cannot be
reliably inferred from geometry alone.

## Files To Customize Next

- Bedroom/tabletop USD path: `BEDROOM_TABLETOP_USD` in `so101_bench_env_cfg.py`
- Tabletop collision subtree: `BEDROOM_TABLETOP_COLLISION_PRIM` in `so101_bench_env_cfg.py`
- Real object mesh registry: `OBJECT_SPLITS` in `so101_bench/benchmark.py`
  plus matching USD files in `source/so101_bench/so101_bench/assets/usd/objects`.
- Camera key mapping for your GR00T fine-tune: `--rename_map` in
  `scripts/groot_eval.py`
- Wrist camera match to your real setup: `INNOMAKER_WRIST_CAMERA_HORIZONTAL_FOV_DEG`,
  `INNOMAKER_WRIST_CAMERA_POS`, and `INNOMAKER_WRIST_CAMERA_RPY_DEG` in
  `source/so101_bench/so101_bench/tasks/direct/so101_bench/so101_bench_env_cfg.py`.

## Some commands

python gr00t/eval/run_gr00t_server.py   --model-path ~/workspace/so101_GR00T-N1.6-3B_WM_v6_55k/checkpoint-55000/   --embodiment-tag NEW_EMBODIMENT   --device cuda   --host 127.0.0.1   --port 5555

gsettings set org.gnome.mutter check-alive-timeout 0

/home/truman/IsaacLab/isaaclab.sh -p scripts/groot_eval.py   --task So101Bench-Bin-v0   --episodes_jsonl tasks/real_gr00t_seen_1obj_WM.jsonl   --policy_host localhost   --policy_port 5555   --action_horizon 16   --use_overhead_init true

/home/truman/IsaacLab/isaaclab.sh -p scripts/view_wrist_camera.py   --task So101Bench-Bin-v0   --display

/home/truman/IsaacLab/isaaclab.sh -p scripts/zero_agent.py --task So101Bench-Bin-v0 --enable_cameras --device cpu
