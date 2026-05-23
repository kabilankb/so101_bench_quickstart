"""Deterministic tabletop layout generation for SO-101 Bench episodes."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any

from so101_bench.benchmark import BenchmarkEpisodeSpec, INCH

MIN_BIN_SURFACE_DISTANCE_M = 1.0 * INCH
DEFAULT_OBJECT_FOOTPRINT_HALF_EXTENTS = (0.02, 0.02)
PLASTIC_BIN_FOOTPRINT_SIZE_IN = (12.675, 7.25)
DEFAULT_BIN_FOOTPRINT_HALF_EXTENTS = (
    0.5 * PLASTIC_BIN_FOOTPRINT_SIZE_IN[0] * INCH,
    0.5 * PLASTIC_BIN_FOOTPRINT_SIZE_IN[1] * INCH,
)
DEFAULT_FOOTPRINT_CENTER_OFFSET = (0.0, 0.0)
DEFAULT_LAYOUT_MAX_ATTEMPTS = 300
DEFAULT_LAYOUT_CANDIDATES_PER_OBJECT = 96
_EPS = 1.0e-9

Point2D = tuple[float, float]
Polygon2D = list[Point2D]


@dataclass(frozen=True)
class _PlacedFootprint:
    object_id: int
    center: Point2D
    yaw: float
    half_extents: Point2D
    center_offset: Point2D
    vertices: Polygon2D

    @property
    def radius(self) -> float:
        return math.hypot(self.half_extents[0], self.half_extents[1])


class LayoutGenerationError(RuntimeError):
    """Raised when a collision-free episode layout cannot be sampled."""


def _xy_polygon(region: list[tuple[float, ...]] | tuple[tuple[float, ...], ...]) -> Polygon2D:
    polygon = [(float(point[0]), float(point[1])) for point in region]
    if len(polygon) < 3:
        raise ValueError(f"Expected a polygon with at least 3 points, got {len(polygon)}.")
    return polygon


def _point_on_segment(point: Point2D, start: Point2D, end: Point2D) -> bool:
    px, py = point
    ax, ay = start
    bx, by = end
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > _EPS:
        return False
    return min(ax, bx) - _EPS <= px <= max(ax, bx) + _EPS and min(ay, by) - _EPS <= py <= max(ay, by) + _EPS


def _point_in_polygon(point: Point2D, polygon: Polygon2D) -> bool:
    inside = False
    px, py = point
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        if _point_on_segment(point, start, end):
            return True
        ax, ay = start
        bx, by = end
        crosses_ray = (ay > py) != (by > py)
        if crosses_ray:
            x_at_y = (bx - ax) * (py - ay) / (by - ay) + ax
            if px < x_at_y:
                inside = not inside
    return inside


def _sample_point_in_polygon(polygon: Polygon2D, rng: random.Random) -> Point2D:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    for _ in range(1000):
        candidate = (rng.uniform(x_min, x_max), rng.uniform(y_min, y_max))
        if _point_in_polygon(candidate, polygon):
            return candidate
    raise LayoutGenerationError("Could not sample a point inside the valid object spawn polygon.")


def _oriented_rectangle(center: Point2D, half_extents: Point2D, yaw: float) -> Polygon2D:
    cx, cy = center
    hx, hy = half_extents
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    return [
        (cx + cos_yaw * dx - sin_yaw * dy, cy + sin_yaw * dx + cos_yaw * dy)
        for dx, dy in corners
    ]


def _rotate_xy(point: Point2D, yaw: float) -> Point2D:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (
        cos_yaw * point[0] - sin_yaw * point[1],
        sin_yaw * point[0] + cos_yaw * point[1],
    )


def _footprint_center(root_center: Point2D, center_offset: Point2D, yaw: float) -> Point2D:
    offset_x, offset_y = _rotate_xy(center_offset, yaw)
    return (root_center[0] + offset_x, root_center[1] + offset_y)


def _footprint_vertices(
    root_center: Point2D,
    half_extents: Point2D,
    center_offset: Point2D,
    yaw: float,
) -> Polygon2D:
    return _oriented_rectangle(_footprint_center(root_center, center_offset, yaw), half_extents, yaw)


def _orientation(a: Point2D, b: Point2D, c: Point2D) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a: Point2D, b: Point2D, c: Point2D, d: Point2D) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)

    if abs(o1) <= _EPS and _point_on_segment(c, a, b):
        return True
    if abs(o2) <= _EPS and _point_on_segment(d, a, b):
        return True
    if abs(o3) <= _EPS and _point_on_segment(a, c, d):
        return True
    if abs(o4) <= _EPS and _point_on_segment(b, c, d):
        return True
    return (o1 > 0.0) != (o2 > 0.0) and (o3 > 0.0) != (o4 > 0.0)


def _point_segment_distance(point: Point2D, start: Point2D, end: Point2D) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= _EPS:
        return math.dist(point, start)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    projection = (ax + t * dx, ay + t * dy)
    return math.dist(point, projection)


def _polygon_contains_any_vertex(subject: Polygon2D, container: Polygon2D) -> bool:
    return any(_point_in_polygon(point, container) for point in subject)


def polygon_surface_distance(first: Polygon2D, second: Polygon2D) -> float:
    """Return the minimum 2D surface distance between two polygon footprints."""

    for first_index, first_start in enumerate(first):
        first_end = first[(first_index + 1) % len(first)]
        for second_index, second_start in enumerate(second):
            second_end = second[(second_index + 1) % len(second)]
            if _segments_intersect(first_start, first_end, second_start, second_end):
                return 0.0

    if _polygon_contains_any_vertex(first, second) or _polygon_contains_any_vertex(second, first):
        return 0.0

    min_distance = math.inf
    for point in first:
        for index, start in enumerate(second):
            min_distance = min(min_distance, _point_segment_distance(point, start, second[(index + 1) % len(second)]))
    for point in second:
        for index, start in enumerate(first):
            min_distance = min(min_distance, _point_segment_distance(point, start, first[(index + 1) % len(first)]))
    return min_distance


def _coerce_half_extents(raw_half_extents: tuple[float, float] | list[float] | None) -> Point2D:
    if raw_half_extents is None:
        return DEFAULT_OBJECT_FOOTPRINT_HALF_EXTENTS
    if len(raw_half_extents) < 2:
        return DEFAULT_OBJECT_FOOTPRINT_HALF_EXTENTS
    return (max(float(raw_half_extents[0]), 0.002), max(float(raw_half_extents[1]), 0.002))


def _coerce_center_offset(raw_center_offset: tuple[float, float] | list[float] | None) -> Point2D:
    if raw_center_offset is None or len(raw_center_offset) < 2:
        return DEFAULT_FOOTPRINT_CENTER_OFFSET
    return (float(raw_center_offset[0]), float(raw_center_offset[1]))


def _coerce_footprint(
    raw_footprint: Any,
    default_half_extents: Point2D,
    default_center_offset: Point2D = DEFAULT_FOOTPRINT_CENTER_OFFSET,
) -> tuple[Point2D, Point2D]:
    if isinstance(raw_footprint, dict):
        half_extents = _coerce_half_extents(
            raw_footprint.get("half_extents") or raw_footprint.get("footprint_half_extents") or default_half_extents
        )
        center_offset = _coerce_center_offset(
            raw_footprint.get("center_offset")
            or raw_footprint.get("footprint_center_offset")
            or default_center_offset
        )
        return half_extents, center_offset
    return _coerce_half_extents(raw_footprint or default_half_extents), default_center_offset


def _coerce_bin_clearance_margin(raw_footprint: Any) -> float:
    if not isinstance(raw_footprint, dict):
        return 0.0
    margin = raw_footprint.get("bin_clearance_margin_m", raw_footprint.get("bin_clearance_margin", 0.0))
    try:
        return max(float(margin), 0.0)
    except (TypeError, ValueError):
        return 0.0


def _entry_yaw(entry: dict[str, Any]) -> float:
    if "yaw" in entry:
        return float(entry["yaw"])
    rpy = entry.get("rpy")
    if isinstance(rpy, (list, tuple)) and len(rpy) >= 3:
        return float(rpy[2])
    raise ValueError(f"Layout object entry is missing a yaw/rpy rotation: {entry!r}.")


def _layout_bin_vertices(
    layout: dict[str, Any],
    bin_footprint_half_extents: tuple[float, float] | list[float] = DEFAULT_BIN_FOOTPRINT_HALF_EXTENTS,
) -> Polygon2D:
    bin_entry = layout.get("bin")
    if not isinstance(bin_entry, dict):
        raise ValueError("Layout is missing a bin entry.")
    bin_position = bin_entry.get("position")
    bin_rpy = bin_entry.get("rpy")
    if not isinstance(bin_position, (list, tuple)) or len(bin_position) < 2:
        raise ValueError(f"Layout bin entry is missing an XY position: {bin_entry!r}.")
    if not isinstance(bin_rpy, (list, tuple)) or len(bin_rpy) < 3:
        raise ValueError(f"Layout bin entry is missing an RPY rotation: {bin_entry!r}.")
    default_bin_half_extents, default_bin_center_offset = _coerce_footprint(
        bin_footprint_half_extents,
        DEFAULT_BIN_FOOTPRINT_HALF_EXTENTS,
    )
    if "center_offset" in bin_entry or "footprint_center_offset" in bin_entry:
        bin_half_extents, bin_center_offset = _coerce_footprint(
            bin_entry,
            default_bin_half_extents,
            default_bin_center_offset,
        )
    else:
        bin_half_extents, bin_center_offset = default_bin_half_extents, default_bin_center_offset
    return _footprint_vertices(
        (float(bin_position[0]), float(bin_position[1])),
        bin_half_extents,
        bin_center_offset,
        float(bin_rpy[2]),
    )


def _layout_object_vertices(
    layout: dict[str, Any],
    object_footprint_half_extents: dict[str, Any],
    object_names_by_slot: list[str] | tuple[str, ...] | None = None,
) -> list[Polygon2D]:
    object_entries = layout.get("objects")
    if not isinstance(object_entries, list) or not object_entries:
        raise ValueError("Layout is missing object entries.")

    object_vertices = []
    for entry in object_entries:
        position = entry.get("position")
        object_name = str(entry.get("name", ""))
        if not object_name and object_names_by_slot is not None and "slot" in entry:
            object_id = int(entry["slot"])
            if 0 <= object_id < len(object_names_by_slot):
                object_name = object_names_by_slot[object_id]
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            raise ValueError(f"Layout object entry is missing an XY position: {entry!r}.")
        default_half_extents, default_center_offset = _coerce_footprint(
            object_footprint_half_extents.get(object_name),
            DEFAULT_OBJECT_FOOTPRINT_HALF_EXTENTS,
        )
        if "center_offset" in entry or "footprint_center_offset" in entry:
            half_extents, center_offset = _coerce_footprint(entry, default_half_extents, default_center_offset)
        else:
            half_extents, center_offset = default_half_extents, default_center_offset
        object_vertices.append(
            _footprint_vertices(
                (float(position[0]), float(position[1])),
                half_extents,
                center_offset,
                _entry_yaw(entry),
            )
        )
    return object_vertices


def layout_bin_surface_distances(
    layout: dict[str, Any],
    object_footprint_half_extents: dict[str, Any],
    bin_footprint_half_extents: Any = DEFAULT_BIN_FOOTPRINT_HALF_EXTENTS,
    object_names_by_slot: list[str] | tuple[str, ...] | None = None,
) -> list[float]:
    """Compute each object's top-down surface distance to the bin footprint."""

    bin_vertices = _layout_bin_vertices(layout, bin_footprint_half_extents)
    object_vertices = _layout_object_vertices(layout, object_footprint_half_extents, object_names_by_slot)
    return [polygon_surface_distance(vertices, bin_vertices) for vertices in object_vertices]


def normalize_layout_object_slots(
    layout: dict[str, Any],
    object_names_by_slot: list[str] | tuple[str, ...],
    *,
    episode_index: int | None = None,
) -> dict[str, Any]:
    """Remap named layout entries to the episode's semantic object slots."""

    object_entries = layout.get("objects")
    if not isinstance(object_entries, list):
        return layout

    named_entries = [entry for entry in object_entries if isinstance(entry, dict) and entry.get("name")]
    if not named_entries:
        return layout

    entry_names = [str(entry["name"]) for entry in named_entries]
    prefix = "Layout"
    if episode_index is not None:
        prefix = f"Layout for episode {episode_index}"
    if len(set(entry_names)) != len(entry_names):
        raise ValueError(f"{prefix} has duplicate object names: {entry_names}.")
    if set(entry_names) != set(object_names_by_slot):
        raise ValueError(
            f"{prefix} does not match episode objects. Expected {list(object_names_by_slot)}, got {entry_names}."
        )

    slot_by_name = {object_name: object_id for object_id, object_name in enumerate(object_names_by_slot)}
    normalized_entries = []
    for entry in object_entries:
        if not isinstance(entry, dict):
            normalized_entries.append(entry)
            continue
        normalized_entry = dict(entry)
        object_name = str(normalized_entry.get("name", ""))
        if object_name:
            slot = slot_by_name[object_name]
            normalized_entry["slot"] = slot
            normalized_entry["asset_name"] = f"object_{slot + 1}"
        normalized_entries.append(normalized_entry)

    normalized_layout = dict(layout)
    normalized_layout["objects"] = sorted(
        normalized_entries,
        key=lambda entry: int(entry.get("slot", 0)) if isinstance(entry, dict) else 0,
    )
    placement = normalized_layout.get("placement")
    if isinstance(placement, dict):
        required_distances = [
            entry.get("required_bin_surface_distance_m")
            for entry in normalized_layout["objects"]
            if isinstance(entry, dict) and "required_bin_surface_distance_m" in entry
        ]
        if len(required_distances) == len(normalized_layout["objects"]):
            normalized_placement = dict(placement)
            normalized_placement["required_bin_surface_distance_m_by_object"] = required_distances
            normalized_layout["placement"] = normalized_placement
    return normalized_layout


def layout_min_bin_surface_distance(
    layout: dict[str, Any],
    object_footprint_half_extents: dict[str, Any],
    bin_footprint_half_extents: Any = DEFAULT_BIN_FOOTPRINT_HALF_EXTENTS,
    object_names_by_slot: list[str] | tuple[str, ...] | None = None,
) -> float:
    """Compute the minimum top-down surface distance from any object footprint to the bin footprint."""

    return min(
        layout_bin_surface_distances(
            layout,
            object_footprint_half_extents,
            bin_footprint_half_extents,
            object_names_by_slot,
        )
    )


def _minimum_surface_distance(footprints: list[_PlacedFootprint]) -> float | None:
    if len(footprints) < 2:
        return None
    min_distance = math.inf
    for left_index, left in enumerate(footprints):
        for right in footprints[left_index + 1 :]:
            min_distance = min(min_distance, polygon_surface_distance(left.vertices, right.vertices))
    return min_distance


def _minimum_bin_surface_distance(footprints: list[_PlacedFootprint], bin_vertices: Polygon2D) -> float:
    return min(polygon_surface_distance(footprint.vertices, bin_vertices) for footprint in footprints)


def _bin_clearance_ok(
    footprints: list[_PlacedFootprint],
    bin_vertices: Polygon2D,
    bin_clearance_m: float,
    object_bin_clearance_margins: list[float],
) -> bool:
    return all(
        polygon_surface_distance(footprint.vertices, bin_vertices)
        >= bin_clearance_m + object_bin_clearance_margins[footprint.object_id]
        for footprint in footprints
    )


def _score_candidate(
    candidate: _PlacedFootprint,
    placed: list[_PlacedFootprint],
    bin_vertices: Polygon2D,
    bin_clearance_m: float,
) -> float:
    bin_distance = polygon_surface_distance(candidate.vertices, bin_vertices)
    if bin_distance < bin_clearance_m:
        return -math.inf

    if not placed:
        return bin_distance

    absolute_distances = []
    for existing in placed:
        surface_distance = polygon_surface_distance(candidate.vertices, existing.vertices)
        absolute_distances.append(surface_distance)
    return min(absolute_distances) + 0.05 * bin_distance


def _sample_spaced_footprints(
    object_footprints: list[tuple[Point2D, Point2D]],
    object_bin_clearance_margins: list[float],
    polygon: Polygon2D,
    bin_vertices: Polygon2D,
    rng: random.Random,
    bin_clearance_m: float,
    max_attempts: int,
    candidates_per_object: int,
) -> tuple[list[_PlacedFootprint], int, int, float, float]:
    count = len(object_footprints)
    base_placement_order = sorted(
        range(count),
        key=lambda object_id: math.hypot(*object_footprints[object_id][0]),
        reverse=True,
    )

    best_footprints: list[_PlacedFootprint] | None = None
    best_min_distance = -math.inf
    best_bin_distance = -math.inf
    valid_attempts = 0
    for attempt in range(1, max_attempts + 1):
        placement_order = list(base_placement_order)
        rng.shuffle(placement_order)
        yaws = [rng.uniform(-math.pi, math.pi) for _ in range(count)]
        placed: list[_PlacedFootprint] = []
        failed = False

        for object_id in placement_order:
            best_candidate: _PlacedFootprint | None = None
            best_score = -math.inf
            for _ in range(candidates_per_object):
                center = _sample_point_in_polygon(polygon, rng)
                half_extents, center_offset = object_footprints[object_id]
                candidate = _PlacedFootprint(
                    object_id=object_id,
                    center=center,
                    yaw=yaws[object_id],
                    half_extents=half_extents,
                    center_offset=center_offset,
                    vertices=_footprint_vertices(center, half_extents, center_offset, yaws[object_id]),
                )
                score = _score_candidate(
                    candidate,
                    placed,
                    bin_vertices,
                    bin_clearance_m + object_bin_clearance_margins[object_id],
                )
                if score > best_score:
                    best_candidate = candidate
                    best_score = score

            if best_candidate is None or not math.isfinite(best_score):
                failed = True
                break
            placed.append(best_candidate)

        if failed:
            continue

        footprints_by_id = sorted(placed, key=lambda footprint: footprint.object_id)
        if not _bin_clearance_ok(footprints_by_id, bin_vertices, bin_clearance_m, object_bin_clearance_margins):
            continue
        min_distance = _minimum_surface_distance(footprints_by_id)
        bin_distance = _minimum_bin_surface_distance(footprints_by_id, bin_vertices)
        if min_distance is None:
            continue
        valid_attempts += 1
        if min_distance > best_min_distance or (
            math.isclose(min_distance, best_min_distance) and bin_distance > best_bin_distance
        ):
            best_footprints = footprints_by_id
            best_min_distance = min_distance
            best_bin_distance = bin_distance

    if best_footprints is not None:
        return best_footprints, max_attempts, valid_attempts, best_min_distance, best_bin_distance

    raise LayoutGenerationError(
        f"Could not sample {count} object placements with >={bin_clearance_m:.5f} m bin surface clearance "
        f"after {max_attempts} attempts."
    )


def _single_object_footprint(
    object_footprint: tuple[Point2D, Point2D],
    object_bin_clearance_margin_m: float,
    polygon: Polygon2D,
    bin_vertices: Polygon2D,
    rng: random.Random,
    bin_clearance_m: float,
    max_attempts: int,
) -> tuple[_PlacedFootprint, int, int, float]:
    best_footprint = None
    best_bin_distance = -math.inf
    valid_attempts = 0
    for attempt in range(1, max_attempts + 1):
        center = _sample_point_in_polygon(polygon, rng)
        yaw = rng.uniform(-math.pi, math.pi)
        half_extents, center_offset = object_footprint
        footprint = _PlacedFootprint(
            object_id=0,
            center=center,
            yaw=yaw,
            half_extents=half_extents,
            center_offset=center_offset,
            vertices=_footprint_vertices(center, half_extents, center_offset, yaw),
        )
        bin_distance = polygon_surface_distance(footprint.vertices, bin_vertices)
        if bin_distance < bin_clearance_m + object_bin_clearance_margin_m:
            continue
        valid_attempts += 1
        if bin_distance > best_bin_distance:
            best_footprint = footprint
            best_bin_distance = bin_distance

    if best_footprint is not None:
        return best_footprint, max_attempts, valid_attempts, best_bin_distance

    raise LayoutGenerationError(
        f"Could not sample single-object placement with "
        f">={bin_clearance_m + object_bin_clearance_margin_m:.5f} m bin surface clearance after "
        f"{max_attempts} attempts."
    )


def generate_episode_layout(
    episode: BenchmarkEpisodeSpec,
    *,
    episode_index: int,
    rng: random.Random,
    bin_random_poses: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...],
    valid_spawn_regions: list[list[tuple[float, float, float]]],
    object_footprint_half_extents: dict[str, Any],
    table_object_z: float,
    seed: int,
    generated_at: str,
    bin_footprint_half_extents: Any = DEFAULT_BIN_FOOTPRINT_HALF_EXTENTS,
    bin_clearance_m: float = MIN_BIN_SURFACE_DISTANCE_M,
    max_attempts: int = DEFAULT_LAYOUT_MAX_ATTEMPTS,
    candidates_per_object: int = DEFAULT_LAYOUT_CANDIDATES_PER_OBJECT,
) -> dict[str, Any]:
    """Sample a replayable initial scene layout for one episode."""

    if len(bin_random_poses) != len(valid_spawn_regions):
        raise ValueError(
            f"Expected one valid object spawn region per bin pose, got "
            f"{len(valid_spawn_regions)} regions for {len(bin_random_poses)} bin poses."
        )
    if not episode.objects:
        raise ValueError("Cannot generate a layout for an episode with no objects.")

    bin_pose_index = rng.randrange(len(bin_random_poses))
    bin_translation, bin_rpy = bin_random_poses[bin_pose_index]
    bin_half_extents, bin_center_offset = _coerce_footprint(
        bin_footprint_half_extents,
        DEFAULT_BIN_FOOTPRINT_HALF_EXTENTS,
    )
    bin_vertices = _footprint_vertices(
        (float(bin_translation[0]), float(bin_translation[1])),
        bin_half_extents,
        bin_center_offset,
        float(bin_rpy[2]),
    )
    polygon = _xy_polygon(valid_spawn_regions[bin_pose_index])
    raw_object_footprints = [object_footprint_half_extents.get(object_name) for object_name in episode.objects]
    object_footprints = [
        _coerce_footprint(raw_footprint, DEFAULT_OBJECT_FOOTPRINT_HALF_EXTENTS)
        for raw_footprint in raw_object_footprints
    ]
    object_bin_clearance_margins = [
        _coerce_bin_clearance_margin(raw_footprint) for raw_footprint in raw_object_footprints
    ]

    if len(episode.objects) == 1:
        footprint, attempts, valid_attempts, min_bin_surface_distance_m = _single_object_footprint(
            object_footprints[0],
            object_bin_clearance_margins[0],
            polygon,
            bin_vertices,
            rng,
            bin_clearance_m,
            max_attempts,
        )
        footprints = [footprint]
        min_between_object_surface_distance_m = None
    else:
        (
            footprints,
            attempts,
            valid_attempts,
            min_between_object_surface_distance_m,
            min_bin_surface_distance_m,
        ) = _sample_spaced_footprints(
            object_footprints,
            object_bin_clearance_margins,
            polygon,
            bin_vertices,
            rng,
            bin_clearance_m,
            max_attempts,
            candidates_per_object,
        )

    metadata = dict(episode.metadata or {})
    objects = []
    for footprint in footprints:
        object_name = episode.objects[footprint.object_id]
        x, y = footprint.center
        required_bin_surface_distance_m = bin_clearance_m + object_bin_clearance_margins[footprint.object_id]
        objects.append(
            {
                "slot": footprint.object_id,
                "asset_name": f"object_{footprint.object_id + 1}",
                "name": object_name,
                "position": [x, y, float(table_object_z)],
                "yaw": footprint.yaw,
                "rpy": [0.0, 0.0, footprint.yaw],
                "footprint_half_extents": [footprint.half_extents[0], footprint.half_extents[1]],
                "footprint_center_offset": [footprint.center_offset[0], footprint.center_offset[1]],
                "required_bin_surface_distance_m": required_bin_surface_distance_m,
            }
        )

    return {
        "trial_id": metadata.get("trial_id", episode_index),
        "episode_index": episode_index,
        "seed": seed,
        "generated_at": generated_at,
        "task_family": episode.task_family,
        "instruction": episode.instruction,
        "objects": objects,
        "bin": {
            "pose_index": bin_pose_index,
            "position": [float(bin_translation[0]), float(bin_translation[1]), float(bin_translation[2])],
            "rpy": [float(bin_rpy[0]), float(bin_rpy[1]), float(bin_rpy[2])],
            "rpy_deg": [math.degrees(float(angle)) for angle in bin_rpy],
            "footprint_half_extents": [bin_half_extents[0], bin_half_extents[1]],
            "footprint_center_offset": [bin_center_offset[0], bin_center_offset[1]],
        },
        "placement": {
            "attempts": attempts,
            "valid_attempts": valid_attempts,
            "min_between_object_surface_distance_m": min_between_object_surface_distance_m,
            "min_surface_distance_m": min_between_object_surface_distance_m,
            "min_bin_surface_distance_m": min_bin_surface_distance_m,
            "required_min_bin_surface_distance_m": bin_clearance_m,
            "required_bin_surface_distance_m_by_object": [
                bin_clearance_m + margin for margin in object_bin_clearance_margins
            ],
            "valid_spawn_region_index": bin_pose_index,
            "max_attempts": max_attempts,
            "candidates_per_object": candidates_per_object if len(episode.objects) > 1 else None,
        },
    }
