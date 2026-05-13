# SO-101 Bench Isaac Lab Environment

This extension turns the Isaac Lab template into an SO-101 Bench environment for
language-conditioned tabletop manipulation with a GR00T-N1.6 fine-tune. It uses
an orange SO-101 arm, a constant plastic bin, one to four tabletop objects, wrist
and overhead RGB-D cameras, and the four benchmark task families from the paper:

- `So101Bench-Bin-v0`: place each object, or the single object, in the plastic bin.
- `So101Bench-NextTo-v0`: place one object next to another.
- `So101Bench-Between-v0`: place one object between two referents.
- `So101Bench-Move-v0`: move one object in a commanded direction.
- `So101Bench-Mixed-v0`: sample among all four families.

The environment currently includes procedural proxy objects and a procedural
fallback tabletop so it can boot before your scanned assets are wired in.

## Required Local Asset Path

The bundled scan at
`source/so101_bench/so101_bench/assets/usd/room_scan.usdc` is used by default
when present. Set this only if you want to use a different bedroom/tabletop USD:

```bash
export SO101_BENCH_BEDROOM_TABLETOP_USD="/absolute/path/to/your/bedroom_tabletop.usd"
```

For the bundled scan, `room_scan/collision_plane/collision_plane` is treated as
the only collidable tabletop mesh and the scanned room mesh is visual-only. The
sim uses the collision plane bounds to scale the scan so the tabletop's short
side is 20 inches in the real world. Override that calibration value, or the
tabletop surface height, if needed:

```bash
export SO101_BENCH_TABLETOP_SHORT_SIDE_IN="20"
```

If only one descendant prim in the scan should collide, set its path relative to
`/BedroomTabletop`. This disables collision on other mesh descendants and enables
it on the selected prim:

```bash
export SO101_BENCH_TABLETOP_COLLISION_PRIM="room_scan/collision_plane/collision_plane"
```

If `SO101_BENCH_BEDROOM_TABLETOP_USD` is not set, the environment uses a simple
kinematic cuboid tabletop at the same height.

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

## GR00T Remote Inference

Start your GR00T-N1.6 fine-tuned policy server separately, then run:

```bash
/home/truman/IsaacLab/isaaclab.sh -p scripts/groot_eval.py \
  --task So101Bench-Bin-v0 \
  --policy_host localhost \
  --policy_port 5555 \
  --num_episodes 10
```

For WM-conditioned checkpoints that expect the real-robot `overhead_init`
camera stream, enable the fixed settled-frame input. The evaluator holds the
initial robot pose for 1 second by default, then records `overhead_init` and
starts querying GR00T:

```bash
/home/truman/IsaacLab/isaaclab.sh -p scripts/groot_eval.py \
  --task So101Bench-Bin-v0 \
  --policy_host localhost \
  --policy_port 5555 \
  --action_horizon 16 \
  --use_overhead_init true
```

By default, the script uses the generated instruction from each environment
reset. To force a fixed instruction:

```bash
/home/truman/IsaacLab/isaaclab.sh -p scripts/groot_eval.py \
  --task So101Bench-NextTo-v0 \
  --lang_instruction "Place the blue bowl next to the black pen."
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

- maximum of three grasp attempts;
- plastic bin displaced by more than 1 inch;
- non-target object displaced by more than 0.25 inches;
- move-task boundary object displaced by more than 0.5 inches;
- bin, next-to, between, and move success geometry with 2-inch tolerances.

Qualitative appendix labels such as semantic error, bad grasp strategy,
occlusion-induced grasp failure, and failed reorientation remain annotation
categories. They are preserved in `so101_bench/benchmark.py` but cannot be
reliably inferred from geometry alone.

## Files To Customize Next

- Bedroom/tabletop USD path: `SO101_BENCH_BEDROOM_TABLETOP_USD`
- Tabletop collision subtree: `SO101_BENCH_TABLETOP_COLLISION_PRIM`
- Real object mesh replacement: replace the four `object_*` proxy assets in
  `source/so101_bench/so101_bench/tasks/direct/so101_bench/so101_bench_env_cfg.py`
  or add your object USD registry there.
- Camera key mapping for your GR00T fine-tune: `--rename_map` in
  `scripts/groot_eval.py`
- Wrist camera match to your real setup: `INNOMAKER_WRIST_CAMERA_HORIZONTAL_FOV_DEG`,
  `INNOMAKER_WRIST_CAMERA_POS`, and `INNOMAKER_WRIST_CAMERA_RPY_DEG` in
  `source/so101_bench/so101_bench/tasks/direct/so101_bench/so101_bench_env_cfg.py`.

## Some commands

python gr00t/eval/run_gr00t_server.py   --model-path ~/workspace/so101_GR00T-N1.6-3B_WM_v6_55k/checkpoint-55000/   --embodiment-tag NEW_EMBODIMENT   --device cuda   --host 127.0.0.1   --port 5555

gsettings set org.gnome.mutter check-alive-timeout 0

/home/truman/IsaacLab/isaaclab.sh -p scripts/groot_eval.py   --task So101Bench-Bin-v0   --policy_host localhost   --policy_port 5555   --action_horizon 16   --use_overhead_init true

/home/truman/IsaacLab/isaaclab.sh -p scripts/view_wrist_camera.py   --task So101Bench-Bin-v0   --display

/home/truman/IsaacLab/isaaclab.sh -p scripts/zero_agent.py --task So101Bench-Bin-v0 --enable_cameras --device cpu
