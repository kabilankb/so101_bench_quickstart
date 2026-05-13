"""Task, object, and evaluation constants for SO-101 Bench.

The task definitions and failure rules are taken from the attached SO-101 Bench
paper and its appendix. Distances are stored in meters for direct use in Isaac Lab.
"""

from __future__ import annotations

INCH = 0.0254

TASK_BIN = "bin"
TASK_NEXT_TO = "next_to"
TASK_BETWEEN = "between"
TASK_MOVE = "move"
TASK_MIXED = "mixed"

TASK_FAMILIES = (TASK_BIN, TASK_NEXT_TO, TASK_BETWEEN, TASK_MOVE)

DIRECTIONS = ("left", "right", "forward", "backward")

MAX_GRASP_ATTEMPTS = 3
GRASP_ATTEMPT_EE_MOTION_M = 1.0 * INCH
BIN_DISPLACEMENT_LIMIT_M = 1.0 * INCH
NON_TARGET_DISPLACEMENT_LIMIT_M = 0.25 * INCH
BOUNDARY_DISPLACEMENT_LIMIT_M = 0.5 * INCH
SPATIAL_SUCCESS_DISTANCE_M = 2.0 * INCH

OBJECT_SPLITS: dict[str, tuple[str, ...]] = {
    "seen": (
        "black glasses",
        "silver glasses",
        "white pen",
        "black pen",
        "altoids container",
        "brown stuffed animal",
        "blue pliers",
        "green clip",
        "pink eraser",
        "yellow wires",
        "grey wires",
        "black screwdriver",
        "yellow screwdriver",
        "red tape",
        "black tape",
        "cardboard box",
        "flower pot",
        "cooking spoon",
        "yellow toy car",
        "grey toy car",
        "green shoes",
        "black shoes",
        "blue bowl",
        "blue scissors",
    ),
    "unseen_seen_class": (
        "orange glasses",
        "white glasses",
        "blue clip",
        "blue tape",
        "yellow tape",
        "white stuffed animal",
        "blue screwdriver",
        "pink bowl",
        "white bowl",
        "black wires",
        "brown wires",
        "orange toy car",
        "blue pen",
        "red pen",
        "white shoes",
    ),
    "unseen_unseen_class": (
        "blue headband",
        "blue highlighter",
        "purple toothbrush",
        "blue controller",
        "action figure",
        "razor",
        "silver tongs",
        "playing cards",
        "candy bar",
        "toy fire truck",
        "toy monster truck",
        "toy dinosaur",
        "baby doll",
        "sponge",
        "yellow flashlight",
    ),
}

BENCHMARK_OBJECT_NAMES: tuple[str, ...] = tuple(
    object_name for split in OBJECT_SPLITS.values() for object_name in split
)

FAILURE_TAXONOMY: dict[str, tuple[str, ...]] = {
    "shared_grasp_acquisition": (
        "bad grasp strategy",
        "imprecise grasp",
        "grabbed air",
        "refused to lift",
        "occlusion-induced grasp failure",
        "got stuck on top",
        "dropped/pushed out of range",
    ),
    "bin_placement": (
        "knocked bin",
        "missed bin",
        "not fully in bin",
    ),
    "shared_instruction_following": (
        "repeatedly reached then docked",
        "semantic error",
        "grasped class distractor",
        "grasped color distractor",
        "grasped other object",
        "failure to reset",
        "wrong task",
        "failed to undock",
        "moved an object",
    ),
    "next_to": (
        "placed next to other object",
        "placed next to class distractor",
        "placed next to color distractor",
        "placed on top",
        "not close enough",
        "drove/rammed into object",
    ),
    "between": (
        "semantic error",
        "placed on top",
        "not centered enough",
        "too close to referent",
        "not close",
    ),
    "move": (
        "not close enough to boundary",
        "not straight enough",
        "moved boundary",
        "moved past boundary",
    ),
}


def task_instruction(task_family: str, active_labels: list[str], direction: str = "") -> str:
    """Return the natural-language instruction for a benchmark episode."""

    if task_family == TASK_BIN:
        if len(active_labels) == 1:
            return f"Place the {active_labels[0]} in the plastic bin."
        return "Place each object in the plastic bin."
    if task_family == TASK_NEXT_TO:
        return f"Place the {active_labels[0]} next to the {active_labels[1]}."
    if task_family == TASK_BETWEEN:
        return f"Place the {active_labels[0]} between the {active_labels[1]} and the {active_labels[2]}."
    if task_family == TASK_MOVE:
        assert direction
        return f"Move the {active_labels[0]} {direction}."
    raise ValueError(f"Unknown task family: {task_family}")

