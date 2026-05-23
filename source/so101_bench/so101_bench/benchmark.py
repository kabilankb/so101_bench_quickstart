"""Task, object, and evaluation constants for SO-101 Bench.

The task definitions and failure rules are taken from the attached SO-101 Bench
paper and its appendix. Distances are stored in meters for direct use in Isaac Lab.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

INCH = 0.0254
MULTI_RIGID_BODY_CHILD_NAMES = ("left", "right")

TASK_BIN = "bin"
TASK_NEXT_TO = "next_to"
TASK_BETWEEN = "between"
TASK_MOVE = "move"
TASK_MIXED = "mixed"

TASK_FAMILIES = (TASK_BIN, TASK_NEXT_TO, TASK_BETWEEN, TASK_MOVE)

DIRECTIONS = ("left", "right", "forward", "backward")

MAX_GRASP_ATTEMPTS = 3
GRASP_ATTEMPT_OBJECT_DISTANCE_M = 4.0 * INCH
BIN_DISPLACEMENT_LIMIT_M = 1.0 * INCH
NON_TARGET_DISPLACEMENT_LIMIT_M = 0.5 * INCH
BOUNDARY_DISPLACEMENT_LIMIT_M = 0.5 * INCH
SPATIAL_SUCCESS_DISTANCE_M = 2.0 * INCH
BETWEEN_LINE_TOLERANCE_M = 1.5 * INCH
ON_TOP_VERTICAL_TOLERANCE_M = 0.5 * INCH

# Each object has an indication for whether its USD contains multiple rigid bodies
# In that case, the object cannot be initialized with RigidObjectCfg and must use AssetBaseCfg instead
OBJECT_SPLITS: dict[str, dict[str, dict[str, bool]]] = {
    "seen": {
        "black glasses": {"multiple_rigid_bodies": False},
        "silver glasses": {"multiple_rigid_bodies": False},
        "white pen": {"multiple_rigid_bodies": False},
        "black pen": {"multiple_rigid_bodies": False},
        "altoids container": {"multiple_rigid_bodies": False},
        # "brown stuffed animal": {"multiple_rigid_bodies": False},  # DO NOT REMOVE
        "blue pliers": {"multiple_rigid_bodies": False},
        "green clip": {"multiple_rigid_bodies": False},
        "pink eraser": {"multiple_rigid_bodies": False},
        "yellow wires": {"multiple_rigid_bodies": False},
        "grey wires": {"multiple_rigid_bodies": False},
        "black screwdriver": {"multiple_rigid_bodies": False},
        "yellow screwdriver": {"multiple_rigid_bodies": False},
        "red tape": {"multiple_rigid_bodies": False},
        "black tape": {"multiple_rigid_bodies": False},
        "cardboard box": {"multiple_rigid_bodies": False},
        "flower pot": {"multiple_rigid_bodies": False},
        "cooking spoon": {"multiple_rigid_bodies": False},
        "yellow toy car": {"multiple_rigid_bodies": False},
        "grey toy car": {"multiple_rigid_bodies": False},
        "green shoes": {"multiple_rigid_bodies": True},
        "black shoes": {"multiple_rigid_bodies": True},
        "blue bowl": {"multiple_rigid_bodies": False},
        "blue scissors": {"multiple_rigid_bodies": False},
    },
    "unseen_seen_class": {
        "orange glasses": {"multiple_rigid_bodies": False},
        "white glasses": {"multiple_rigid_bodies": False},
        "blue clip": {"multiple_rigid_bodies": False},
        "blue tape": {"multiple_rigid_bodies": False},
        "yellow tape": {"multiple_rigid_bodies": False},
        # "white stuffed animal", DO NOT REMOVE
        "blue screwdriver": {"multiple_rigid_bodies": False},
        "pink bowl": {"multiple_rigid_bodies": False},
        "white bowl": {"multiple_rigid_bodies": False},
        "black wires": {"multiple_rigid_bodies": False},
        "brown wires": {"multiple_rigid_bodies": False},
        "orange toy car": {"multiple_rigid_bodies": False},
        "blue pen": {"multiple_rigid_bodies": False},
        "red pen": {"multiple_rigid_bodies": False},
        "white shoes": {"multiple_rigid_bodies": True},
    },
    "unseen_unseen_class": {
        # "blue headband", DO NOT REMOVE
        "blue highlighter": {"multiple_rigid_bodies": False},
        "purple toothbrush": {"multiple_rigid_bodies": False},
        "blue controller": {"multiple_rigid_bodies": False},
        "action figure": {"multiple_rigid_bodies": False},
        "razor": {"multiple_rigid_bodies": False},
        "silver tongs": {"multiple_rigid_bodies": False},
        "playing cards": {"multiple_rigid_bodies": False},
        "candy bar": {"multiple_rigid_bodies": False},
        "toy fire truck": {"multiple_rigid_bodies": False},
        "toy monster truck": {"multiple_rigid_bodies": False},
        "toy dinosaur": {"multiple_rigid_bodies": False},
        # "baby doll", DO NOT REMOVE
        "sponge": {"multiple_rigid_bodies": False},
        "yellow flashlight": {"multiple_rigid_bodies": False},
    },
}

BENCHMARK_OBJECT_NAMES: tuple[str, ...] = tuple(
    object_name for split in OBJECT_SPLITS.values() for object_name in split
)
OBJECT_METADATA: dict[str, dict[str, bool]] = {
    object_name: metadata for split in OBJECT_SPLITS.values() for object_name, metadata in split.items()
}

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


@dataclass(frozen=True)
class BenchmarkEpisodeSpec:
    """Validated JSONL episode metadata consumed by reset and scene configuration."""

    objects: tuple[str, ...]
    instruction: str
    task_family: str
    target_object_id: int
    referent_object_ids: tuple[int, int]
    direction: str | None = None
    metadata: dict[str, Any] | None = None

    def reset_payload(self) -> dict[str, Any]:
        """Return the JSON-compatible subset needed by the reset event."""

        return {
            "objects": list(self.objects),
            "instruction": self.instruction,
            "task_family": self.task_family,
            "active_object_ids": list(range(len(self.objects))),
            "target_object_id": self.target_object_id,
            "referent_object_ids": list(self.referent_object_ids),
            "direction": self.direction,
            "metadata": dict(self.metadata or {}),
        }


def object_metadata(object_name: str) -> dict[str, bool]:
    """Return validated metadata for an object name from ``OBJECT_SPLITS``."""

    try:
        return OBJECT_METADATA[object_name]
    except KeyError as exc:
        valid = ", ".join(BENCHMARK_OBJECT_NAMES)
        raise ValueError(f"Unknown benchmark object {object_name!r}. Expected one of: {valid}.") from exc


def object_rigid_body_child_names(object_name: str) -> tuple[str, ...]:
    """Return child prim names that should be treated as independently movable rigid bodies."""

    if object_metadata(object_name)["multiple_rigid_bodies"]:
        return MULTI_RIGID_BODY_CHILD_NAMES
    return ()


def object_usd_stem(object_name: str) -> str:
    """Return the local object USD filename stem used by JSONL labels."""

    object_metadata(object_name)
    return object_name.replace(" ", "_")


def _normalized_instruction(instruction: str) -> str:
    return " ".join(instruction.strip().lower().rstrip(".").split())


def _canonical_direction(token: str) -> str:
    direction = token.lower()
    aliases = {"forwards": "forward", "backwards": "backward"}
    direction = aliases.get(direction, direction)
    if direction not in DIRECTIONS:
        raise ValueError(f"Unknown move direction {token!r}. Expected one of: {', '.join(DIRECTIONS)}.")
    return direction


def infer_task_family(instruction: str) -> str:
    """Infer one of the four benchmark task families from an instruction."""

    normalized = _normalized_instruction(instruction)
    if "plastic bin" in normalized and normalized.startswith("place"):
        return TASK_BIN
    if normalized.startswith("place") and " next to " in normalized:
        return TASK_NEXT_TO
    if normalized.startswith("place") and " between " in normalized and " and " in normalized:
        return TASK_BETWEEN
    if normalized.startswith("move "):
        return TASK_MOVE
    raise ValueError(
        f"Instruction {instruction!r} does not match a supported benchmark task. "
        "Expected bin, next-to, between, or directional move phrasing."
    )


def _object_mentions(instruction: str, objects: tuple[str, ...]) -> list[tuple[int, int]]:
    mentions: list[tuple[int, int]] = []
    lowered_instruction = instruction.lower()
    for object_id, object_name in enumerate(objects):
        match = re.search(rf"(?<!\w){re.escape(object_name.lower())}(?!\w)", lowered_instruction)
        if match is not None:
            mentions.append((match.start(), object_id))
    return sorted(mentions)


def _object_id_from_row_name(objects: tuple[str, ...], row: dict[str, Any], key: str) -> int | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or value not in objects:
        raise ValueError(f"JSONL field {key!r} must name one of the row objects, got {value!r}.")
    return objects.index(value)


def _referent_ids_from_row(objects: tuple[str, ...], row: dict[str, Any]) -> list[int]:
    referents = row.get("referents")
    if referents is None:
        return []
    if not isinstance(referents, list) or not all(isinstance(name, str) for name in referents):
        raise ValueError(f"JSONL field 'referents' must be a list of object names, got {referents!r}.")
    invalid = [name for name in referents if name not in objects]
    if invalid:
        raise ValueError(f"JSONL referents must be present in 'objects', got invalid referents: {invalid}.")
    return [objects.index(name) for name in referents]


def _referent_pair(object_count: int, referent_ids: list[int], fallback: list[int]) -> tuple[int, int]:
    ids = [*referent_ids, *fallback]
    if not ids:
        return (0, 0)
    first = ids[0]
    second = ids[1] if len(ids) > 1 else first
    for object_id in (first, second):
        if object_id < 0 or object_id >= object_count:
            raise ValueError(f"Referent object id {object_id} is out of range for {object_count} objects.")
    return (first, second)


def episode_spec_from_json(row: dict[str, Any], *, source: str = "JSONL row") -> BenchmarkEpisodeSpec:
    """Validate a JSONL row and derive task indices used by the simulator."""

    raw_objects = row.get("objects")
    if not isinstance(raw_objects, list) or not raw_objects or not all(isinstance(name, str) for name in raw_objects):
        raise ValueError(f"{source}: 'objects' must be a non-empty list of benchmark object names.")
    objects = tuple(raw_objects)
    if len(objects) > 4:
        raise ValueError(f"{source}: the benchmark supports at most four tabletop objects, got {len(objects)}.")
    if len(set(objects)) != len(objects):
        raise ValueError(f"{source}: episode object names must be unique, got {list(objects)}.")
    for object_name in objects:
        object_metadata(object_name)

    n_objects = row.get("n_objects")
    if n_objects is not None and n_objects != len(objects):
        raise ValueError(f"{source}: n_objects={n_objects!r} does not match {len(objects)} objects.")

    instruction = row.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"{source}: 'instruction' must be a non-empty string.")
    instruction = instruction.strip()
    inferred_task_family = infer_task_family(instruction)
    task_family = row.get("task_family") or inferred_task_family
    if task_family not in TASK_FAMILIES:
        raise ValueError(f"{source}: unsupported task_family {task_family!r}.")
    if task_family != inferred_task_family:
        raise ValueError(
            f"{source}: task_family {task_family!r} does not match the instruction family "
            f"{inferred_task_family!r}."
        )

    mentions = [object_id for _offset, object_id in _object_mentions(instruction, objects)]
    row_target_id = _object_id_from_row_name(objects, row, "target")
    row_referent_ids = _referent_ids_from_row(objects, row)

    if task_family == TASK_BIN:
        if len(objects) not in {1, 4}:
            raise ValueError(f"{source}: bin episodes must contain either one or four objects.")
        target_id = 0
        referents = _referent_pair(len(objects), [], mentions[1:])
        direction = None
    elif task_family == TASK_NEXT_TO:
        if len(objects) != 4:
            raise ValueError(f"{source}: next-to episodes must contain four objects.")
        if row_target_id is None and len(mentions) < 2:
            raise ValueError(f"{source}: next-to instruction must mention target and referent objects.")
        target_id = row_target_id if row_target_id is not None else mentions[0]
        fallback = [object_id for object_id in mentions if object_id != target_id]
        referents = _referent_pair(len(objects), row_referent_ids, fallback)
        if referents[0] == target_id:
            raise ValueError(f"{source}: next-to episodes need distinct target and referent objects.")
        direction = None
    elif task_family == TASK_BETWEEN:
        if len(objects) != 4:
            raise ValueError(f"{source}: between episodes must contain four objects.")
        if row_target_id is None and len(mentions) < 3:
            raise ValueError(f"{source}: between instruction must mention target and two referent objects.")
        target_id = row_target_id if row_target_id is not None else mentions[0]
        fallback = [object_id for object_id in mentions if object_id != target_id]
        referents = _referent_pair(len(objects), row_referent_ids, fallback)
        if referents[0] == referents[1] or target_id in referents:
            raise ValueError(f"{source}: between episodes need a target and two distinct referents.")
        direction = None
    else:
        if row_target_id is None and not mentions:
            raise ValueError(f"{source}: move instruction must mention the moved object.")
        target_id = row_target_id if row_target_id is not None else mentions[0]
        direction_value = row.get("direction")
        if direction_value is None:
            match = re.search(r"\b(left|right|forwards?|backwards?)\b", instruction, flags=re.IGNORECASE)
            if match is None:
                raise ValueError(f"{source}: move instruction must include a direction.")
            direction_value = match.group(1)
        if not isinstance(direction_value, str):
            raise ValueError(f"{source}: move direction must be a string, got {direction_value!r}.")
        direction = _canonical_direction(direction_value)
        fallback = [object_id for object_id in range(len(objects)) if object_id != target_id]
        referents = _referent_pair(len(objects), row_referent_ids, fallback)

    return BenchmarkEpisodeSpec(
        objects=objects,
        instruction=instruction,
        task_family=task_family,
        target_object_id=target_id,
        referent_object_ids=referents,
        direction=direction,
        metadata={key: value for key, value in row.items() if key not in {"objects", "instruction"}},
    )


def load_episode_jsonl(path: str | Path) -> list[BenchmarkEpisodeSpec]:
    """Load and validate benchmark episodes from a JSONL file."""

    path = Path(path)
    episodes: list[BenchmarkEpisodeSpec] = []
    with path.open(encoding="utf-8") as jsonl_file:
        for line_no, line in enumerate(jsonl_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}.") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object per line.")
            episodes.append(episode_spec_from_json(row, source=f"{path}:{line_no}"))
    if not episodes:
        raise ValueError(f"{path}: JSONL file did not contain any benchmark episodes.")
    return episodes
