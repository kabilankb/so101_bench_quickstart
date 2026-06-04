"""Deterministic tabletop layout generation for SO-101 Bench episodes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
import math
import random
from typing import Any

from so101_bench.benchmark import (
    BETWEEN_LINE_TOLERANCE_M,
    BenchmarkEpisodeSpec,
    DIRECTIONS,
    INCH,
    MOVE_BOUNDARY_MIN_LATERAL_OVERLAP_FRACTION,
    MOVE_STRAIGHTNESS_TOLERANCE_M,
    SPATIAL_SUCCESS_DISTANCE_M,
    TASK_BETWEEN,
    TASK_BIN,
    TASK_MOVE,
    TASK_NEXT_TO,
    load_object_move_footprint_boxes,
)

MIN_BIN_SURFACE_DISTANCE_M = 0.5 * INCH
MIN_OBJECT_SURFACE_DISTANCE_M = 0.0
MIN_INITIAL_TABLE_BOUNDS_INSIDE_FRACTION = 0.75
# Pad the measured base outline so footprint approximation error cannot admit near-contact poses.
MIN_ROBOT_SURFACE_DISTANCE_M = 0.25 * INCH
DEFAULT_OBJECT_FOOTPRINT_HALF_EXTENTS = (0.02, 0.02)
PLASTIC_BIN_FOOTPRINT_SIZE_IN = (12.675, 7.25)
DEFAULT_BIN_FOOTPRINT_HALF_EXTENTS = (
    0.5 * PLASTIC_BIN_FOOTPRINT_SIZE_IN[0] * INCH,
    0.5 * PLASTIC_BIN_FOOTPRINT_SIZE_IN[1] * INCH,
)
DEFAULT_FOOTPRINT_CENTER_OFFSET = (0.0, 0.0)
DEFAULT_LAYOUT_MAX_ATTEMPTS = 144
DEFAULT_LAYOUT_CANDIDATES_PER_OBJECT = 48
DEFAULT_TOP_VALID_LAYOUT_CANDIDATES = 16
NEXT_TO_FEASIBILITY_YAW_STEPS = 16
NEXT_TO_FEASIBILITY_ANGLE_STEPS = 32
NEXT_TO_FEASIBILITY_GAP_FRACTIONS = (0.02, 0.125, 0.25, 0.5, 1.0)
BETWEEN_FEASIBILITY_YAW_STEPS = 16
BETWEEN_FEASIBILITY_SEGMENT_STEPS = 11
BETWEEN_FEASIBILITY_OFFSET_STEPS = 10
BETWEEN_FEASIBILITY_MIN_SEGMENT_FRACTION = 0.18
BETWEEN_FEASIBILITY_MAX_MIDPOINT_OFFSET_FRACTION = 0.15
BETWEEN_FEASIBILITY_MIN_TARGET_TRAVEL_M = 3.0 * INCH
MOVE_FEASIBILITY_MIN_BOUNDARY_GAP_M = 3.5 * INCH
_DIRECTIONAL_GAP_LATERAL_STEP_M = 0.005
_MOVE_BOUNDARY_SUGGESTED_GAP_OFFSETS_M = (0.0, 0.25 * INCH, 0.5 * INCH, 1.0 * INCH)
_MOVE_BOUNDARY_SUGGESTED_LATERAL_FRACTIONS = (0.0, -0.2, 0.2, -0.4, 0.4)
_EPS = 1.0e-9

Point2D = tuple[float, float]
Polygon2D = list[Point2D]
MoveFootprintBox = tuple[float, float, float, float]
MoveFootprintBoxes = tuple[MoveFootprintBox, ...]
TableBounds = dict[str, tuple[float, float] | list[float]]
LayoutConstraint = Callable[[list["_PlacedFootprint"]], dict[str, Any] | None]
CandidateScoreAdjustment = Callable[[list["_PlacedFootprint"]], float]
CandidateCenterSuggestions = Callable[[int, float, list["_PlacedFootprint"]], list[Point2D]]


@dataclass(frozen=True)
class _PlacedFootprint:
    object_id: int
    center: Point2D
    yaw: float
    half_extents: Point2D
    center_offset: Point2D
    vertices: Polygon2D
    move_footprint_boxes: MoveFootprintBoxes = ()

    @property
    def radius(self) -> float:
        return math.hypot(self.half_extents[0], self.half_extents[1])

    @cached_property
    def move_vertices(self) -> list[Polygon2D]:
        if not self.move_footprint_boxes:
            return [self.vertices]
        return _move_footprint_piece_vertices(self.center, self.yaw, self.move_footprint_boxes)


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


def _move_footprint_piece_vertices(
    root_center: Point2D,
    yaw: float,
    boxes: MoveFootprintBoxes,
) -> list[Polygon2D]:
    return [
        [
            (
                root_center[0] + _rotate_xy((local_x, local_y), yaw)[0],
                root_center[1] + _rotate_xy((local_x, local_y), yaw)[1],
            )
            for local_x, local_y in (
                (box[0], box[1]),
                (box[2], box[1]),
                (box[2], box[3]),
                (box[0], box[3]),
            )
        ]
        for box in boxes
    ]


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


def _segment_polygon_surface_distance(start: Point2D, end: Point2D, polygon: Polygon2D) -> float:
    if _point_in_polygon(start, polygon) or _point_in_polygon(end, polygon):
        return 0.0
    if any(
        _segments_intersect(start, end, polygon_start, polygon[(index + 1) % len(polygon)])
        for index, polygon_start in enumerate(polygon)
    ):
        return 0.0
    return min(_point_segment_distance(point, start, end) for point in polygon)


def _aabb_surface_distance(first: Polygon2D, second: Polygon2D) -> float:
    first_xs = [point[0] for point in first]
    first_ys = [point[1] for point in first]
    second_xs = [point[0] for point in second]
    second_ys = [point[1] for point in second]
    dx = max(min(second_xs) - max(first_xs), min(first_xs) - max(second_xs), 0.0)
    dy = max(min(second_ys) - max(first_ys), min(first_ys) - max(second_ys), 0.0)
    return math.hypot(dx, dy)


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


def _coerce_move_footprint_boxes(raw_boxes: Any) -> MoveFootprintBoxes:
    if not isinstance(raw_boxes, (list, tuple)):
        return ()
    boxes = []
    for raw_box in raw_boxes:
        if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
            return ()
        box = tuple(float(value) for value in raw_box)
        if not all(math.isfinite(value) for value in box) or box[0] >= box[2] or box[1] >= box[3]:
            return ()
        boxes.append(box)
    return tuple(boxes)


def _object_move_footprint_boxes(object_name: str, raw_footprint: Any) -> MoveFootprintBoxes:
    if isinstance(raw_footprint, dict):
        boxes = _coerce_move_footprint_boxes(raw_footprint.get("move_footprint_boxes"))
        if boxes:
            return boxes
    try:
        return load_object_move_footprint_boxes(object_name, required=False)
    except ValueError:
        return ()


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


def _layout_placed_footprints(
    layout: dict[str, Any],
    object_footprint_half_extents: dict[str, Any],
    object_names_by_slot: list[str] | tuple[str, ...] | None = None,
) -> list[_PlacedFootprint]:
    object_entries = layout.get("objects")
    if not isinstance(object_entries, list) or not object_entries:
        raise ValueError("Layout is missing object entries.")

    footprints = []
    for fallback_object_id, entry in enumerate(object_entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Layout object entry must be a dict, got {entry!r}.")
        position = entry.get("position")
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            raise ValueError(f"Layout object entry is missing an XY position: {entry!r}.")

        object_id = int(entry.get("slot", fallback_object_id))
        object_name = str(entry.get("name", ""))
        if not object_name and object_names_by_slot is not None and 0 <= object_id < len(object_names_by_slot):
            object_name = object_names_by_slot[object_id]

        default_half_extents, default_center_offset = _coerce_footprint(
            object_footprint_half_extents.get(object_name),
            DEFAULT_OBJECT_FOOTPRINT_HALF_EXTENTS,
        )
        if "center_offset" in entry or "footprint_center_offset" in entry:
            half_extents, center_offset = _coerce_footprint(entry, default_half_extents, default_center_offset)
        else:
            half_extents, center_offset = default_half_extents, default_center_offset
        center = (float(position[0]), float(position[1]))
        yaw = _entry_yaw(entry)
        raw_footprint = object_footprint_half_extents.get(object_name)
        footprints.append(
            _PlacedFootprint(
                object_id=object_id,
                center=center,
                yaw=yaw,
                half_extents=half_extents,
                center_offset=center_offset,
                vertices=_footprint_vertices(center, half_extents, center_offset, yaw),
                move_footprint_boxes=_object_move_footprint_boxes(object_name, raw_footprint),
            )
        )
    return sorted(footprints, key=lambda footprint: footprint.object_id)


def _placed_footprint(
    object_id: int,
    center: Point2D,
    yaw: float,
    half_extents: Point2D,
    center_offset: Point2D,
    move_footprint_boxes: MoveFootprintBoxes = (),
) -> _PlacedFootprint:
    return _PlacedFootprint(
        object_id=object_id,
        center=center,
        yaw=yaw,
        half_extents=half_extents,
        center_offset=center_offset,
        vertices=_footprint_vertices(center, half_extents, center_offset, yaw),
        move_footprint_boxes=move_footprint_boxes,
    )


def _target_at_footprint_center(target: _PlacedFootprint, footprint_center: Point2D, yaw: float) -> _PlacedFootprint:
    offset_x, offset_y = _rotate_xy(target.center_offset, yaw)
    root_center = (footprint_center[0] - offset_x, footprint_center[1] - offset_y)
    return _placed_footprint(
        target.object_id,
        root_center,
        yaw,
        target.half_extents,
        target.center_offset,
        target.move_footprint_boxes,
    )


def _normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _unique_angles(angles: list[float]) -> list[float]:
    unique: list[float] = []
    for angle in angles:
        normalized = _normalize_angle(float(angle))
        if all(abs(_normalize_angle(normalized - existing)) > 1.0e-6 for existing in unique):
            unique.append(normalized)
    return unique


def _next_to_candidate_yaws(target: _PlacedFootprint, referent: _PlacedFootprint) -> list[float]:
    uniform = [
        2.0 * math.pi * yaw_index / NEXT_TO_FEASIBILITY_YAW_STEPS
        for yaw_index in range(NEXT_TO_FEASIBILITY_YAW_STEPS)
    ]
    anchored = [
        target.yaw,
        referent.yaw,
        referent.yaw + 0.5 * math.pi,
        referent.yaw + math.pi,
        referent.yaw + 1.5 * math.pi,
    ]
    return _unique_angles([*uniform, *anchored])


def _table_bounds_contain(vertices: Polygon2D, table_bounds: TableBounds) -> bool:
    x_min, x_max = table_bounds["x"]
    y_min, y_max = table_bounds["y"]
    return all(
        float(x_min) - _EPS <= x <= float(x_max) + _EPS
        and float(y_min) - _EPS <= y <= float(y_max) + _EPS
        for x, y in vertices
    )


def _polygon_area(vertices: Polygon2D) -> float:
    return 0.5 * abs(
        sum(
            start[0] * end[1] - end[0] * start[1]
            for start, end in zip(vertices, vertices[1:] + vertices[:1], strict=True)
        )
    )


def _clip_polygon_to_axis_bound(
    vertices: Polygon2D,
    *,
    axis: int,
    bound: float,
    keep_greater: bool,
) -> Polygon2D:
    if not vertices:
        return []

    def inside(point: Point2D) -> bool:
        return point[axis] >= bound - _EPS if keep_greater else point[axis] <= bound + _EPS

    clipped: Polygon2D = []
    for start, end in zip(vertices, vertices[1:] + vertices[:1], strict=True):
        start_inside = inside(start)
        end_inside = inside(end)
        if start_inside != end_inside:
            delta = end[axis] - start[axis]
            t = 0.0 if abs(delta) <= _EPS else (bound - start[axis]) / delta
            clipped.append(
                (
                    start[0] + t * (end[0] - start[0]),
                    start[1] + t * (end[1] - start[1]),
                )
            )
        if end_inside:
            clipped.append(end)
    return clipped


def _table_bounds_contain_fraction(
    vertices: Polygon2D,
    table_bounds: TableBounds,
    *,
    min_inside_fraction: float,
) -> bool:
    clipped = vertices
    for axis, bound, keep_greater in (
        (0, float(table_bounds["x"][0]), True),
        (0, float(table_bounds["x"][1]), False),
        (1, float(table_bounds["y"][0]), True),
        (1, float(table_bounds["y"][1]), False),
    ):
        clipped = _clip_polygon_to_axis_bound(
            clipped,
            axis=axis,
            bound=bound,
            keep_greater=keep_greater,
        )
    footprint_area = _polygon_area(vertices)
    return footprint_area > _EPS and _polygon_area(clipped) + _EPS >= min_inside_fraction * footprint_area


def _next_to_final_pose_metrics(
    candidate: _PlacedFootprint,
    footprints_by_id: dict[int, _PlacedFootprint],
    referent_object_id: int,
    bin_vertices: Polygon2D,
    table_bounds: TableBounds,
    success_distance_m: float,
    min_object_surface_distance_m: float,
    robot_bounding_box: Polygon2D | None,
    robot_clearance_m: float,
) -> dict[str, Any] | None:
    if not _table_bounds_contain(candidate.vertices, table_bounds):
        return None

    referent = footprints_by_id[referent_object_id]
    referent_distance = polygon_surface_distance(candidate.vertices, referent.vertices)
    if referent_distance <= min_object_surface_distance_m + _EPS:
        return None
    if referent_distance > success_distance_m + _EPS:
        return None

    bin_distance = polygon_surface_distance(candidate.vertices, bin_vertices)
    if bin_distance <= _EPS:
        return None

    if robot_bounding_box is not None:
        robot_distance = polygon_surface_distance(candidate.vertices, robot_bounding_box)
        if robot_distance < robot_clearance_m - _EPS:
            return None
    else:
        robot_distance = None

    min_other_distance = math.inf
    for object_id, existing in footprints_by_id.items():
        if object_id == candidate.object_id or object_id == referent_object_id:
            continue
        other_distance = polygon_surface_distance(candidate.vertices, existing.vertices)
        if other_distance <= min_object_surface_distance_m + _EPS:
            return None
        min_other_distance = min(min_other_distance, other_distance)

    return {
        "feasible_target_position": [candidate.center[0], candidate.center[1]],
        "feasible_target_yaw": candidate.yaw,
        "feasible_target_referent_surface_distance_m": referent_distance,
        "feasible_target_bin_surface_distance_m": bin_distance,
        "feasible_target_robot_surface_distance_m": robot_distance,
        "feasible_target_min_other_surface_distance_m": None
        if math.isinf(min_other_distance)
        else min_other_distance,
    }


def _next_to_feasibility(
    footprints: list[_PlacedFootprint],
    target_object_id: int,
    referent_object_id: int,
    bin_vertices: Polygon2D,
    table_bounds: TableBounds,
    success_distance_m: float,
    min_object_surface_distance_m: float,
    robot_bounding_box: Polygon2D | None,
    robot_clearance_m: float,
) -> dict[str, Any] | None:
    footprints_by_id = {footprint.object_id: footprint for footprint in footprints}
    target = footprints_by_id.get(target_object_id)
    referent = footprints_by_id.get(referent_object_id)
    if target is None or referent is None:
        return None

    initial_distance = polygon_surface_distance(target.vertices, referent.vertices)
    initial_aabb_distance = _aabb_surface_distance(target.vertices, referent.vertices)
    if initial_distance <= success_distance_m + _EPS:
        return None
    if initial_aabb_distance <= success_distance_m + _EPS:
        return None

    referent_center = _footprint_center(referent.center, referent.center_offset, referent.yaw)
    max_center_distance = target.radius + referent.radius + success_distance_m + 0.01
    gaps = [max(success_distance_m * fraction, 1.0e-5) for fraction in NEXT_TO_FEASIBILITY_GAP_FRACTIONS]

    for yaw in _next_to_candidate_yaws(target, referent):
        for angle_index in range(NEXT_TO_FEASIBILITY_ANGLE_STEPS):
            angle = 2.0 * math.pi * angle_index / NEXT_TO_FEASIBILITY_ANGLE_STEPS
            unit = (math.cos(angle), math.sin(angle))

            low = 0.0
            high = max_center_distance
            high_candidate = _target_at_footprint_center(
                target,
                (referent_center[0] + unit[0] * high, referent_center[1] + unit[1] * high),
                yaw,
            )
            if polygon_surface_distance(high_candidate.vertices, referent.vertices) <= _EPS:
                continue

            for _ in range(24):
                middle = 0.5 * (low + high)
                candidate = _target_at_footprint_center(
                    target,
                    (referent_center[0] + unit[0] * middle, referent_center[1] + unit[1] * middle),
                    yaw,
                )
                if polygon_surface_distance(candidate.vertices, referent.vertices) <= _EPS:
                    low = middle
                else:
                    high = middle

            for gap in gaps:
                center_distance = high + gap
                if center_distance > max_center_distance:
                    continue
                candidate = _target_at_footprint_center(
                    target,
                    (
                        referent_center[0] + unit[0] * center_distance,
                        referent_center[1] + unit[1] * center_distance,
                    ),
                    yaw,
                )
                metrics = _next_to_final_pose_metrics(
                    candidate,
                    footprints_by_id,
                    referent_object_id,
                    bin_vertices,
                    table_bounds,
                    success_distance_m,
                    min_object_surface_distance_m,
                    robot_bounding_box,
                    robot_clearance_m,
                )
                if metrics is not None:
                    return {
                        "initial_target_referent_surface_distance_m": initial_distance,
                        "initial_target_referent_aabb_surface_distance_m": initial_aabb_distance,
                        "success_distance_m": success_distance_m,
                        **metrics,
                    }
    return None


def layout_next_to_feasibility(
    layout: dict[str, Any],
    object_footprint_half_extents: dict[str, Any],
    bin_footprint_half_extents: Any,
    object_names_by_slot: list[str] | tuple[str, ...],
    target_object_id: int,
    referent_object_id: int,
    table_bounds: TableBounds,
    *,
    success_distance_m: float = SPATIAL_SUCCESS_DISTANCE_M,
    min_object_surface_distance_m: float = MIN_OBJECT_SURFACE_DISTANCE_M,
    robot_bounding_box: list[tuple[float, ...]] | tuple[tuple[float, ...], ...] | None = None,
    robot_clearance_m: float = MIN_ROBOT_SURFACE_DISTANCE_M,
) -> dict[str, Any] | None:
    """Return next-to feasibility metrics for a layout, or ``None`` if no valid final pose exists."""

    footprints = _layout_placed_footprints(layout, object_footprint_half_extents, object_names_by_slot)
    bin_vertices = _layout_bin_vertices(layout, bin_footprint_half_extents)
    robot_polygon = _xy_polygon(robot_bounding_box) if robot_bounding_box is not None else None
    return _next_to_feasibility(
        footprints,
        target_object_id,
        referent_object_id,
        bin_vertices,
        table_bounds,
        max(float(success_distance_m), 0.0),
        max(float(min_object_surface_distance_m), 0.0),
        robot_polygon,
        max(float(robot_clearance_m), 0.0),
    )


def _between_position_metrics(
    point: Point2D,
    referent_a: Point2D,
    referent_b: Point2D,
    centered_tolerance_m: float,
    min_segment_fraction: float,
) -> dict[str, float] | None:
    segment_x = referent_b[0] - referent_a[0]
    segment_y = referent_b[1] - referent_a[1]
    segment_len_sq = segment_x * segment_x + segment_y * segment_y
    if segment_len_sq <= _EPS:
        return None

    point_x = point[0] - referent_a[0]
    point_y = point[1] - referent_a[1]
    fraction = (point_x * segment_x + point_y * segment_y) / segment_len_sq
    projection = (
        referent_a[0] + fraction * segment_x,
        referent_a[1] + fraction * segment_y,
    )
    perpendicular = math.dist(point, projection)
    if fraction < min_segment_fraction - _EPS or fraction > 1.0 - min_segment_fraction + _EPS:
        return None
    if perpendicular > centered_tolerance_m + _EPS:
        return None
    return {
        "target_segment_fraction": fraction,
        "target_perpendicular_distance_m": perpendicular,
    }


def _distance_to_between_success_region(
    point: Point2D,
    referent_a: Point2D,
    referent_b: Point2D,
    centered_tolerance_m: float,
    min_segment_fraction: float,
) -> float:
    segment_x = referent_b[0] - referent_a[0]
    segment_y = referent_b[1] - referent_a[1]
    segment_len_sq = segment_x * segment_x + segment_y * segment_y
    if segment_len_sq <= _EPS:
        return math.inf

    segment_length = math.sqrt(segment_len_sq)
    point_x = point[0] - referent_a[0]
    point_y = point[1] - referent_a[1]
    fraction = (point_x * segment_x + point_y * segment_y) / segment_len_sq
    clamped_fraction = max(min_segment_fraction, min(1.0 - min_segment_fraction, fraction))
    projection = (
        referent_a[0] + fraction * segment_x,
        referent_a[1] + fraction * segment_y,
    )
    perpendicular = math.dist(point, projection)
    parallel_excess = abs(fraction - clamped_fraction) * segment_length
    perpendicular_excess = max(perpendicular - centered_tolerance_m, 0.0)
    return math.hypot(parallel_excess, perpendicular_excess)


def _between_candidate_yaws(
    target: _PlacedFootprint,
    referent_a: _PlacedFootprint,
    referent_b: _PlacedFootprint,
) -> list[float]:
    segment_angle = math.atan2(
        referent_b.center[1] - referent_a.center[1],
        referent_b.center[0] - referent_a.center[0],
    )
    uniform = [
        2.0 * math.pi * yaw_index / BETWEEN_FEASIBILITY_YAW_STEPS
        for yaw_index in range(BETWEEN_FEASIBILITY_YAW_STEPS)
    ]
    anchored = [
        target.yaw,
        referent_a.yaw,
        referent_b.yaw,
        segment_angle,
        segment_angle + 0.5 * math.pi,
        segment_angle + math.pi,
        segment_angle + 1.5 * math.pi,
    ]
    return _unique_angles([*anchored, *uniform])


def _ordered_between_segment_fractions(min_segment_fraction: float) -> list[float]:
    success_lower = max(float(min_segment_fraction), 0.0)
    success_upper = min(1.0 - success_lower, 1.0)
    lower = max(success_lower, 0.5 - BETWEEN_FEASIBILITY_MAX_MIDPOINT_OFFSET_FRACTION)
    upper = min(success_upper, 0.5 + BETWEEN_FEASIBILITY_MAX_MIDPOINT_OFFSET_FRACTION)
    if lower > upper + _EPS:
        return []
    if math.isclose(lower, upper):
        return [0.5 * (lower + upper)]

    fractions = [
        lower + (upper - lower) * index / (BETWEEN_FEASIBILITY_SEGMENT_STEPS - 1)
        for index in range(BETWEEN_FEASIBILITY_SEGMENT_STEPS)
    ]
    fractions.append(0.5 * (lower + upper))
    return sorted(_unique_scalar_values(fractions), key=lambda fraction: (abs(fraction - 0.5), fraction))


def _unique_scalar_values(values: list[float]) -> list[float]:
    unique: list[float] = []
    for value in values:
        if all(abs(value - existing) > 1.0e-9 for existing in unique):
            unique.append(float(value))
    return unique


def _ordered_between_offsets(centered_tolerance_m: float) -> list[float]:
    tolerance = max(float(centered_tolerance_m), 0.0)
    if tolerance <= _EPS or BETWEEN_FEASIBILITY_OFFSET_STEPS <= 1:
        return [0.0]

    positive_steps = max((BETWEEN_FEASIBILITY_OFFSET_STEPS - 1) // 2, 1)
    offsets = [0.0]
    for index in range(1, positive_steps + 1):
        offset = tolerance * index / positive_steps
        offsets.extend((offset, -offset))
    return offsets


def _between_candidate_centers(
    referent_a: _PlacedFootprint,
    referent_b: _PlacedFootprint,
    centered_tolerance_m: float,
    min_segment_fraction: float,
) -> list[Point2D]:
    segment_x = referent_b.center[0] - referent_a.center[0]
    segment_y = referent_b.center[1] - referent_a.center[1]
    segment_length = math.hypot(segment_x, segment_y)
    if segment_length <= _EPS:
        return []

    normal = (-segment_y / segment_length, segment_x / segment_length)
    centers = []
    for fraction in _ordered_between_segment_fractions(min_segment_fraction):
        base = (
            referent_a.center[0] + fraction * segment_x,
            referent_a.center[1] + fraction * segment_y,
        )
        for offset in _ordered_between_offsets(centered_tolerance_m):
            centers.append((base[0] + normal[0] * offset, base[1] + normal[1] * offset))
    return centers


def _between_final_pose_metrics(
    candidate: _PlacedFootprint,
    footprints_by_id: dict[int, _PlacedFootprint],
    referent_object_ids: tuple[int, int],
    bin_vertices: Polygon2D,
    table_bounds: TableBounds,
    centered_tolerance_m: float,
    min_segment_fraction: float,
    min_object_surface_distance_m: float,
    robot_bounding_box: Polygon2D | None,
    robot_clearance_m: float,
) -> dict[str, Any] | None:
    referent_a = footprints_by_id[referent_object_ids[0]]
    referent_b = footprints_by_id[referent_object_ids[1]]
    between_metrics = _between_position_metrics(
        candidate.center,
        referent_a.center,
        referent_b.center,
        centered_tolerance_m,
        min_segment_fraction,
    )
    if between_metrics is None:
        return None

    if not _table_bounds_contain(candidate.vertices, table_bounds):
        return None

    bin_distance = polygon_surface_distance(candidate.vertices, bin_vertices)
    if bin_distance <= _EPS:
        return None

    if robot_bounding_box is not None:
        robot_distance = polygon_surface_distance(candidate.vertices, robot_bounding_box)
        if robot_distance < robot_clearance_m - _EPS:
            return None
    else:
        robot_distance = None

    referent_distances: list[float | None] = [None, None]
    min_object_distance = math.inf
    for object_id, existing in footprints_by_id.items():
        if object_id == candidate.object_id:
            continue
        object_distance = polygon_surface_distance(candidate.vertices, existing.vertices)
        if object_distance <= min_object_surface_distance_m + _EPS:
            return None
        min_object_distance = min(min_object_distance, object_distance)
        if object_id == referent_object_ids[0]:
            referent_distances[0] = object_distance
        if object_id == referent_object_ids[1]:
            referent_distances[1] = object_distance

    return {
        "feasible_target_position": [candidate.center[0], candidate.center[1]],
        "feasible_target_yaw": candidate.yaw,
        "feasible_target_segment_fraction": between_metrics["target_segment_fraction"],
        "feasible_target_perpendicular_distance_m": between_metrics["target_perpendicular_distance_m"],
        "feasible_target_referent_surface_distances_m": referent_distances,
        "feasible_target_bin_surface_distance_m": bin_distance,
        "feasible_target_robot_surface_distance_m": robot_distance,
        "feasible_target_min_object_surface_distance_m": None
        if math.isinf(min_object_distance)
        else min_object_distance,
    }


def _between_feasibility(
    footprints: list[_PlacedFootprint],
    target_object_id: int,
    referent_object_ids: tuple[int, int],
    bin_vertices: Polygon2D,
    table_bounds: TableBounds,
    centered_tolerance_m: float,
    min_segment_fraction: float,
    min_object_surface_distance_m: float,
    robot_bounding_box: Polygon2D | None,
    robot_clearance_m: float,
) -> dict[str, Any] | None:
    footprints_by_id = {footprint.object_id: footprint for footprint in footprints}
    target = footprints_by_id.get(target_object_id)
    referent_a = footprints_by_id.get(referent_object_ids[0])
    referent_b = footprints_by_id.get(referent_object_ids[1])
    if target is None or referent_a is None or referent_b is None:
        return None

    initial_between_metrics = _between_position_metrics(
        target.center,
        referent_a.center,
        referent_b.center,
        centered_tolerance_m,
        min_segment_fraction,
    )
    initial_distance_to_success_region = _distance_to_between_success_region(
        target.center,
        referent_a.center,
        referent_b.center,
        centered_tolerance_m,
        min_segment_fraction,
    )
    if (
        initial_between_metrics is not None
        or initial_distance_to_success_region < BETWEEN_FEASIBILITY_MIN_TARGET_TRAVEL_M - _EPS
    ):
        return None

    if robot_bounding_box is not None:
        referent_segment_robot_distance = _segment_polygon_surface_distance(
            referent_a.center,
            referent_b.center,
            robot_bounding_box,
        )
        if referent_segment_robot_distance < robot_clearance_m - _EPS:
            return None
    else:
        referent_segment_robot_distance = None

    candidate_centers = _between_candidate_centers(
        referent_a,
        referent_b,
        centered_tolerance_m,
        min_segment_fraction,
    )
    candidate_yaws = _between_candidate_yaws(target, referent_a, referent_b)
    for center in candidate_centers:
        for yaw in candidate_yaws:
            candidate = _placed_footprint(
                target.object_id,
                center,
                yaw,
                target.half_extents,
                target.center_offset,
                target.move_footprint_boxes,
            )
            metrics = _between_final_pose_metrics(
                candidate,
                footprints_by_id,
                referent_object_ids,
                bin_vertices,
                table_bounds,
                centered_tolerance_m,
                min_segment_fraction,
                min_object_surface_distance_m,
                robot_bounding_box,
                robot_clearance_m,
            )
            if metrics is not None:
                return {
                    "initial_target_segment_fraction": None,
                    "initial_target_perpendicular_distance_m": None,
                    "initial_target_distance_to_success_region_m": initial_distance_to_success_region,
                    "required_min_initial_target_distance_to_success_region_m": BETWEEN_FEASIBILITY_MIN_TARGET_TRAVEL_M,
                    "centered_tolerance_m": centered_tolerance_m,
                    "min_segment_fraction": min_segment_fraction,
                    "feasibility_max_midpoint_offset_fraction": BETWEEN_FEASIBILITY_MAX_MIDPOINT_OFFSET_FRACTION,
                    "referent_segment_robot_surface_distance_m": referent_segment_robot_distance,
                    **metrics,
                }
    return None


def layout_between_feasibility(
    layout: dict[str, Any],
    object_footprint_half_extents: dict[str, Any],
    bin_footprint_half_extents: Any,
    object_names_by_slot: list[str] | tuple[str, ...],
    target_object_id: int,
    referent_object_ids: tuple[int, int] | list[int],
    table_bounds: TableBounds,
    *,
    centered_tolerance_m: float = BETWEEN_LINE_TOLERANCE_M,
    min_segment_fraction: float = BETWEEN_FEASIBILITY_MIN_SEGMENT_FRACTION,
    min_object_surface_distance_m: float = MIN_OBJECT_SURFACE_DISTANCE_M,
    robot_bounding_box: list[tuple[float, ...]] | tuple[tuple[float, ...], ...] | None = None,
    robot_clearance_m: float = MIN_ROBOT_SURFACE_DISTANCE_M,
) -> dict[str, Any] | None:
    """Return between-task feasibility metrics for a layout, or ``None`` if no valid final pose exists."""

    referents = tuple(int(object_id) for object_id in referent_object_ids)
    if len(referents) != 2:
        raise ValueError(f"Between feasibility requires two referent ids, got {referent_object_ids!r}.")
    footprints = _layout_placed_footprints(layout, object_footprint_half_extents, object_names_by_slot)
    bin_vertices = _layout_bin_vertices(layout, bin_footprint_half_extents)
    robot_polygon = _xy_polygon(robot_bounding_box) if robot_bounding_box is not None else None
    return _between_feasibility(
        footprints,
        target_object_id,
        (referents[0], referents[1]),
        bin_vertices,
        table_bounds,
        max(float(centered_tolerance_m), 0.0),
        max(float(min_segment_fraction), 0.0),
        max(float(min_object_surface_distance_m), 0.0),
        robot_polygon,
        max(float(robot_clearance_m), 0.0),
    )


def _move_direction_axis_and_sign(direction: str) -> tuple[int, float]:
    if direction == "left":
        return (0, 1.0)
    if direction == "right":
        return (0, -1.0)
    if direction == "forward":
        return (1, -1.0)
    if direction == "backward":
        return (1, 1.0)
    raise ValueError(f"Unknown move direction {direction!r}. Expected one of: {', '.join(DIRECTIONS)}.")


def _move_direction_vector(direction: str) -> Point2D:
    axis, sign = _move_direction_axis_and_sign(direction)
    return (sign, 0.0) if axis == 0 else (0.0, sign)


def _piece_vertices_axis_bounds(piece_vertices: list[Polygon2D], axis: int) -> tuple[float, float]:
    values = [point[axis] for vertices in piece_vertices for point in vertices]
    return min(values), max(values)


def _polygon_cross_section_axis_extents(
    vertices: Polygon2D,
    axis: int,
    lateral_axis: int,
    lateral_value: float,
) -> tuple[float, float] | None:
    intersections = []
    for index, start in enumerate(vertices):
        end = vertices[(index + 1) % len(vertices)]
        start_lateral = start[lateral_axis]
        end_lateral = end[lateral_axis]
        delta_lateral = end_lateral - start_lateral
        if abs(delta_lateral) <= _EPS:
            if abs(lateral_value - start_lateral) <= _EPS:
                intersections.extend((start[axis], end[axis]))
            continue
        fraction = (lateral_value - start_lateral) / delta_lateral
        if -_EPS <= fraction <= 1.0 + _EPS:
            intersections.append(start[axis] + fraction * (end[axis] - start[axis]))
    if not intersections:
        return None
    return min(intersections), max(intersections)


def _footprint_cross_section_axis_extents(
    piece_vertices: list[Polygon2D],
    axis: int,
    lateral_axis: int,
    lateral_value: float,
) -> tuple[float, float] | None:
    extents = [
        extent
        for vertices in piece_vertices
        if (extent := _polygon_cross_section_axis_extents(vertices, axis, lateral_axis, lateral_value)) is not None
    ]
    if not extents:
        return None
    return min(extent[0] for extent in extents), max(extent[1] for extent in extents)


def _directional_footprint_gap(
    target_piece_vertices: list[Polygon2D],
    boundary_piece_vertices: list[Polygon2D],
    axis: int,
    sign: float,
) -> float | None:
    lateral_axis = 1 - axis
    target_lateral = [point[lateral_axis] for vertices in target_piece_vertices for point in vertices]
    boundary_lateral = [point[lateral_axis] for vertices in boundary_piece_vertices for point in vertices]
    lateral_lo = max(min(target_lateral), min(boundary_lateral))
    lateral_hi = min(max(target_lateral), max(boundary_lateral))
    if lateral_hi <= lateral_lo + _EPS:
        return None

    samples = min(int((lateral_hi - lateral_lo) / _DIRECTIONAL_GAP_LATERAL_STEP_M) + 2, 256)
    min_gap = math.inf
    for index in range(samples):
        lateral_value = lateral_lo + (lateral_hi - lateral_lo) * index / (samples - 1)
        target_extents = _footprint_cross_section_axis_extents(
            target_piece_vertices,
            axis,
            lateral_axis,
            lateral_value,
        )
        boundary_extents = _footprint_cross_section_axis_extents(
            boundary_piece_vertices,
            axis,
            lateral_axis,
            lateral_value,
        )
        if target_extents is None or boundary_extents is None:
            continue
        target_front = target_extents[1] if sign > 0.0 else target_extents[0]
        boundary_surface = boundary_extents[0] if sign > 0.0 else boundary_extents[1]
        min_gap = min(min_gap, sign * (boundary_surface - target_front))
    return None if math.isinf(min_gap) else min_gap


def _directional_footprint_ahead_extent(
    target_piece_vertices: list[Polygon2D],
    boundary_piece_vertices: list[Polygon2D],
    axis: int,
    sign: float,
) -> float | None:
    lateral_axis = 1 - axis
    target_lateral = [point[lateral_axis] for vertices in target_piece_vertices for point in vertices]
    boundary_lateral = [point[lateral_axis] for vertices in boundary_piece_vertices for point in vertices]
    lateral_lo = max(min(target_lateral), min(boundary_lateral))
    lateral_hi = min(max(target_lateral), max(boundary_lateral))
    if lateral_hi <= lateral_lo + _EPS:
        return None

    samples = min(int((lateral_hi - lateral_lo) / _DIRECTIONAL_GAP_LATERAL_STEP_M) + 2, 256)
    max_ahead = -math.inf
    for index in range(samples):
        lateral_value = lateral_lo + (lateral_hi - lateral_lo) * index / (samples - 1)
        target_extents = _footprint_cross_section_axis_extents(
            target_piece_vertices,
            axis,
            lateral_axis,
            lateral_value,
        )
        boundary_extents = _footprint_cross_section_axis_extents(
            boundary_piece_vertices,
            axis,
            lateral_axis,
            lateral_value,
        )
        if target_extents is None or boundary_extents is None:
            continue
        target_front = target_extents[1] if sign > 0.0 else target_extents[0]
        boundary_far = boundary_extents[1] if sign > 0.0 else boundary_extents[0]
        max_ahead = max(max_ahead, sign * (boundary_far - target_front))
    return None if math.isinf(max_ahead) else max_ahead


def _lateral_overlap_width(
    target_piece_vertices: list[Polygon2D],
    boundary_piece_vertices: list[Polygon2D],
    axis: int,
) -> float:
    lateral_axis = 1 - axis
    target_lateral = [point[lateral_axis] for vertices in target_piece_vertices for point in vertices]
    boundary_lateral = [point[lateral_axis] for vertices in boundary_piece_vertices for point in vertices]
    lateral_lo = max(min(target_lateral), min(boundary_lateral))
    lateral_hi = min(max(target_lateral), max(boundary_lateral))
    if lateral_hi <= lateral_lo + _EPS:
        return 0.0

    samples = min(int((lateral_hi - lateral_lo) / _DIRECTIONAL_GAP_LATERAL_STEP_M) + 2, 256)
    overlapping_samples = 0
    for index in range(samples):
        lateral_value = lateral_lo + (lateral_hi - lateral_lo) * index / (samples - 1)
        if (
            _footprint_cross_section_axis_extents(target_piece_vertices, axis, lateral_axis, lateral_value)
            is not None
            and _footprint_cross_section_axis_extents(
                boundary_piece_vertices,
                axis,
                lateral_axis,
                lateral_value,
            )
            is not None
        ):
            overlapping_samples += 1
    return overlapping_samples / samples * (lateral_hi - lateral_lo)


def _move_footprint_projection_bounds(footprint: _PlacedFootprint, axis: int) -> tuple[float, float]:
    values = [point[axis] for vertices in footprint.move_vertices for point in vertices]
    return min(values), max(values)


def _translated_polygon(vertices: Polygon2D, delta: Point2D) -> Polygon2D:
    return [(point[0] + delta[0], point[1] + delta[1]) for point in vertices]


def _convex_hull(points: list[Point2D]) -> Polygon2D:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return unique

    def cross(origin: Point2D, first: Point2D, second: Point2D) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[Point2D] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= _EPS:
            lower.pop()
        lower.append(point)

    upper: list[Point2D] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= _EPS:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def _swept_polygon(vertices: Polygon2D, delta: Point2D) -> Polygon2D:
    return _convex_hull([*vertices, *_translated_polygon(vertices, delta)])


def _move_swept_piece_vertices(
    target: _PlacedFootprint,
    direction: str,
    distance_m: float,
) -> list[Polygon2D]:
    move_direction = _move_direction_vector(direction)
    delta = (move_direction[0] * distance_m, move_direction[1] * distance_m)
    return [_swept_polygon(vertices, delta) for vertices in target.move_vertices]


def _move_target_forward_table_gap(
    target: _PlacedFootprint,
    direction: str,
    table_bounds: TableBounds,
) -> float:
    axis, sign = _move_direction_axis_and_sign(direction)
    target_axis_min, target_axis_max = _move_footprint_projection_bounds(target, axis)
    far_bound = table_bounds["x" if axis == 0 else "y"][1 if sign > 0.0 else 0]
    target_front = target_axis_max if sign > 0.0 else target_axis_min
    return sign * (float(far_bound) - target_front)


def _move_swept_object_surface_distances(
    target: _PlacedFootprint,
    footprints_by_id: dict[int, _PlacedFootprint],
    direction: str,
    clear_path_m: float,
) -> list[dict[str, Any]]:
    swept_target_vertices = _move_swept_piece_vertices(target, direction, clear_path_m)
    distances = []
    for object_id, footprint in footprints_by_id.items():
        if object_id == target.object_id:
            continue
        min_distance = math.inf
        for swept_vertices in swept_target_vertices:
            for object_vertices in footprint.move_vertices:
                min_distance = min(min_distance, polygon_surface_distance(swept_vertices, object_vertices))
        distances.append({"object_id": object_id, "surface_distance_m": min_distance})
    return distances


def _move_boundary_gaps(
    target: _PlacedFootprint,
    footprints_by_id: dict[int, _PlacedFootprint],
    direction: str,
) -> list[dict[str, Any]]:
    axis, sign = _move_direction_axis_and_sign(direction)
    lateral_axis = 1 - axis
    target_vertices = target.move_vertices
    target_lateral = [point[lateral_axis] for vertices in target_vertices for point in vertices]
    min_lateral_overlap_m = (
        MOVE_BOUNDARY_MIN_LATERAL_OVERLAP_FRACTION * (max(target_lateral) - min(target_lateral))
    )
    gaps: list[dict[str, Any]] = []

    for object_id, footprint in footprints_by_id.items():
        if object_id == target.object_id:
            continue
        boundary_vertices = footprint.move_vertices
        gap = _directional_footprint_gap(target_vertices, boundary_vertices, axis, sign)
        ahead = _directional_footprint_ahead_extent(target_vertices, boundary_vertices, axis, sign)
        if gap is None or ahead is None or ahead <= _EPS:
            continue
        lateral_overlap_m = _lateral_overlap_width(target_vertices, boundary_vertices, axis)
        if lateral_overlap_m < min_lateral_overlap_m - _EPS:
            continue
        gaps.append(
            {
                "boundary_type": "object",
                "boundary_id": object_id,
                "surface_gap_m": gap,
                "lateral_overlap_m": lateral_overlap_m,
                "required_min_lateral_overlap_m": min_lateral_overlap_m,
            }
        )

    return gaps


def _move_boundary_candidate_centers(
    target: _PlacedFootprint,
    object_footprint: tuple[Point2D, Point2D] | tuple[Point2D, Point2D, MoveFootprintBoxes],
    yaw: float,
    direction: str,
    min_boundary_gap_m: float,
) -> list[Point2D]:
    axis, sign = _move_direction_axis_and_sign(direction)
    lateral_axis = 1 - axis
    target_vertices = target.move_vertices
    target_axis_min, target_axis_max = _piece_vertices_axis_bounds(target_vertices, axis)
    target_lateral_min, target_lateral_max = _piece_vertices_axis_bounds(target_vertices, lateral_axis)
    target_front = target_axis_max if sign > 0.0 else target_axis_min
    target_lateral_center = 0.5 * (target_lateral_min + target_lateral_max)
    target_lateral_width = target_lateral_max - target_lateral_min

    half_extents, center_offset, move_footprint_boxes = _sampled_object_footprint_parts(object_footprint)
    if move_footprint_boxes:
        candidate_vertices = _move_footprint_piece_vertices((0.0, 0.0), yaw, move_footprint_boxes)
    else:
        candidate_vertices = [_footprint_vertices((0.0, 0.0), half_extents, center_offset, yaw)]
    candidate_axis_min, candidate_axis_max = _piece_vertices_axis_bounds(candidate_vertices, axis)
    candidate_lateral_min, candidate_lateral_max = _piece_vertices_axis_bounds(candidate_vertices, lateral_axis)
    candidate_near = candidate_axis_min if sign > 0.0 else candidate_axis_max
    candidate_lateral_center = 0.5 * (candidate_lateral_min + candidate_lateral_max)

    centers: list[Point2D] = []
    for gap_offset_m in _MOVE_BOUNDARY_SUGGESTED_GAP_OFFSETS_M:
        desired_near = target_front + sign * (min_boundary_gap_m + gap_offset_m)
        center_axis = desired_near - candidate_near
        for lateral_fraction in _MOVE_BOUNDARY_SUGGESTED_LATERAL_FRACTIONS:
            center = [0.0, 0.0]
            center[axis] = center_axis
            center[lateral_axis] = (
                target_lateral_center
                + lateral_fraction * target_lateral_width
                - candidate_lateral_center
            )
            centers.append((center[0], center[1]))
    return centers


def _move_target_candidate_centers(
    object_footprint: tuple[Point2D, Point2D] | tuple[Point2D, Point2D, MoveFootprintBoxes],
    yaw: float,
    direction: str,
    table_bounds: TableBounds,
    min_boundary_gap_m: float,
) -> list[Point2D]:
    axis, sign = _move_direction_axis_and_sign(direction)
    lateral_axis = 1 - axis
    half_extents, center_offset, move_footprint_boxes = _sampled_object_footprint_parts(object_footprint)
    if move_footprint_boxes:
        target_vertices = _move_footprint_piece_vertices((0.0, 0.0), yaw, move_footprint_boxes)
    else:
        target_vertices = [_footprint_vertices((0.0, 0.0), half_extents, center_offset, yaw)]

    target_axis_min, target_axis_max = _piece_vertices_axis_bounds(target_vertices, axis)
    target_lateral_min, target_lateral_max = _piece_vertices_axis_bounds(target_vertices, lateral_axis)
    target_front = target_axis_max if sign > 0.0 else target_axis_min

    axis_bounds = table_bounds["x" if axis == 0 else "y"]
    lateral_bounds = table_bounds["x" if lateral_axis == 0 else "y"]
    axis_min = float(axis_bounds[0]) - target_axis_min
    axis_max = float(axis_bounds[1]) - target_axis_max
    lateral_min = float(lateral_bounds[0]) - target_lateral_min
    lateral_max = float(lateral_bounds[1]) - target_lateral_max
    if axis_max < axis_min - _EPS or lateral_max < lateral_min - _EPS:
        return []

    far_bound = float(axis_bounds[1 if sign > 0.0 else 0])
    centers: list[Point2D] = []
    axis_steps = 7
    lateral_steps = 7
    for axis_index in range(axis_steps):
        axis_fraction = 0.5 if axis_steps == 1 else axis_index / (axis_steps - 1)
        center_axis = axis_min + axis_fraction * (axis_max - axis_min)
        target_front_world = center_axis + target_front
        if sign * (far_bound - target_front_world) < min_boundary_gap_m - _EPS:
            continue
        for lateral_index in range(lateral_steps):
            lateral_fraction = 0.5 if lateral_steps == 1 else 0.15 + 0.7 * lateral_index / (lateral_steps - 1)
            center = [0.0, 0.0]
            center[axis] = center_axis
            center[lateral_axis] = lateral_min + lateral_fraction * (lateral_max - lateral_min)
            centers.append((center[0], center[1]))
    return centers


def _move_candidate_score_adjustment(
    footprints: list[_PlacedFootprint],
    target_object_id: int,
    direction: str,
    table_bounds: TableBounds,
    min_boundary_gap_m: float,
) -> float:
    footprints_by_id = {footprint.object_id: footprint for footprint in footprints}
    target = footprints_by_id.get(target_object_id)
    if target is None:
        return 0.0
    if not _table_bounds_contain(target.vertices, table_bounds):
        return -math.inf
    if _move_target_forward_table_gap(target, direction, table_bounds) < min_boundary_gap_m - _EPS:
        return -math.inf
    clear_path_m = max(min_boundary_gap_m - 1.0e-6, 0.0)
    if not all(
        _table_bounds_contain(vertices, table_bounds)
        for vertices in _move_swept_piece_vertices(target, direction, clear_path_m)
    ):
        return -math.inf

    axis, sign = _move_direction_axis_and_sign(direction)
    lateral_axis = 1 - axis
    target_vertices = target.move_vertices
    target_lateral_min, target_lateral_max = _piece_vertices_axis_bounds(target_vertices, lateral_axis)
    target_lateral_width = target_lateral_max - target_lateral_min
    min_lateral_overlap_m = MOVE_BOUNDARY_MIN_LATERAL_OVERLAP_FRACTION * target_lateral_width
    target_axis_min, target_axis_max = _piece_vertices_axis_bounds(target_vertices, axis)
    target_front = target_axis_max if sign > 0.0 else target_axis_min

    boundary_score = 0.0
    for object_id, footprint in footprints_by_id.items():
        if object_id == target_object_id:
            continue
        boundary_vertices = footprint.move_vertices
        boundary_lateral_min, boundary_lateral_max = _piece_vertices_axis_bounds(
            boundary_vertices,
            lateral_axis,
        )
        lateral_overlap_m = min(target_lateral_max, boundary_lateral_max) - max(
            target_lateral_min,
            boundary_lateral_min,
        )
        if lateral_overlap_m < min_lateral_overlap_m - _EPS:
            continue
        boundary_axis_min, boundary_axis_max = _piece_vertices_axis_bounds(boundary_vertices, axis)
        boundary_surface = boundary_axis_min if sign > 0.0 else boundary_axis_max
        boundary_far = boundary_axis_max if sign > 0.0 else boundary_axis_min
        gap = sign * (boundary_surface - target_front)
        ahead = sign * (boundary_far - target_front)
        if gap >= min_boundary_gap_m - _EPS and ahead > _EPS:
            boundary_score = max(boundary_score, lateral_overlap_m)
    if boundary_score <= 0.0:
        return 0.0
    return 10.0 + boundary_score


def _move_feasibility(
    footprints: list[_PlacedFootprint],
    target_object_id: int,
    direction: str,
    bin_vertices: Polygon2D,
    table_bounds: TableBounds,
    min_boundary_gap_m: float,
    straightness_tolerance_m: float,
    robot_bounding_box: Polygon2D | None,
) -> dict[str, Any] | None:
    _ = bin_vertices
    footprints_by_id = {footprint.object_id: footprint for footprint in footprints}
    target = footprints_by_id.get(target_object_id)
    if target is None:
        return None

    if not _table_bounds_contain(target.vertices, table_bounds):
        return None

    boundary_gaps = [
        entry
        for entry in _move_boundary_gaps(target, footprints_by_id, direction)
        if _table_bounds_contain_fraction(
            footprints_by_id[int(entry["boundary_id"])].vertices,
            table_bounds,
            min_inside_fraction=MIN_INITIAL_TABLE_BOUNDS_INSIDE_FRACTION,
        )
    ]
    if not boundary_gaps:
        return None
    min_boundary_gap = min(float(entry["surface_gap_m"]) for entry in boundary_gaps)
    if min_boundary_gap < min_boundary_gap_m - _EPS:
        return None

    selected_boundary = max(
        boundary_gaps,
        key=lambda entry: (float(entry["lateral_overlap_m"]), -float(entry["surface_gap_m"])),
    )
    clear_path_m = max(min_boundary_gap_m - 1.0e-6, 0.0)
    forward_table_gap_m = _move_target_forward_table_gap(target, direction, table_bounds)
    if forward_table_gap_m < min_boundary_gap_m - _EPS:
        return None

    swept_target_vertices = _move_swept_piece_vertices(target, direction, clear_path_m)
    if not all(_table_bounds_contain(vertices, table_bounds) for vertices in swept_target_vertices):
        return None

    swept_path_intersects_robot = robot_bounding_box is not None and any(
        polygon_surface_distance(vertices, robot_bounding_box) <= _EPS for vertices in swept_target_vertices
    )
    if swept_path_intersects_robot:
        return None

    swept_object_distances = _move_swept_object_surface_distances(
        target,
        footprints_by_id,
        direction,
        clear_path_m,
    )
    swept_path_blockers = [
        entry for entry in swept_object_distances if float(entry["surface_distance_m"]) <= _EPS
    ]
    if swept_path_blockers:
        return None

    axis, sign = _move_direction_axis_and_sign(direction)
    min_swept_object_distance = min(
        (float(entry["surface_distance_m"]) for entry in swept_object_distances),
        default=None,
    )
    return {
        "direction": direction,
        "axis": axis,
        "sign": sign,
        "straightness_tolerance_m": straightness_tolerance_m,
        "required_min_boundary_gap_m": min_boundary_gap_m,
        "required_min_clear_path_m": clear_path_m,
        "initial_target_min_boundary_gap_m": min_boundary_gap,
        "initial_target_clear_path_m": clear_path_m,
        "initial_target_forward_table_gap_m": forward_table_gap_m,
        "initial_target_boundary_gaps_m": boundary_gaps,
        "initial_target_selected_boundary": selected_boundary,
        "initial_target_boundary_overlap_m": float(selected_boundary["lateral_overlap_m"]),
        "layout_selection_score": float(selected_boundary["lateral_overlap_m"]),
        "initial_target_move_ray_origin": [target.center[0], target.center[1]],
        "initial_target_move_ray_intersects_robot": False,
        "initial_target_swept_path_intersects_robot": False,
        "initial_target_swept_object_surface_distances_m": swept_object_distances,
        "initial_target_min_swept_object_surface_distance_m": min_swept_object_distance,
        "initial_target_swept_path_blockers": [],
        "initial_target_yaw": target.yaw,
    }


def layout_move_feasibility(
    layout: dict[str, Any],
    object_footprint_half_extents: dict[str, Any],
    bin_footprint_half_extents: Any,
    object_names_by_slot: list[str] | tuple[str, ...],
    target_object_id: int,
    direction: str,
    table_bounds: TableBounds,
    *,
    min_boundary_gap_m: float = MOVE_FEASIBILITY_MIN_BOUNDARY_GAP_M,
    straightness_tolerance_m: float = MOVE_STRAIGHTNESS_TOLERANCE_M,
    robot_bounding_box: list[tuple[float, ...]] | tuple[tuple[float, ...], ...] | None = None,
) -> dict[str, Any] | None:
    """Return directional move clearance metrics, or ``None`` if the forward path is blocked."""

    footprints = _layout_placed_footprints(layout, object_footprint_half_extents, object_names_by_slot)
    bin_vertices = _layout_bin_vertices(layout, bin_footprint_half_extents)
    robot_polygon = _xy_polygon(robot_bounding_box) if robot_bounding_box is not None else None
    return _move_feasibility(
        footprints,
        int(target_object_id),
        str(direction),
        bin_vertices,
        table_bounds,
        max(float(min_boundary_gap_m), 0.0),
        max(float(straightness_tolerance_m), 0.0),
        robot_polygon,
    )


def layout_task_feasibility(
    layout: dict[str, Any],
    episode: BenchmarkEpisodeSpec,
    object_footprint_half_extents: dict[str, Any],
    bin_footprint_half_extents: Any,
    table_bounds: TableBounds,
    *,
    robot_bounding_box: list[tuple[float, ...]] | tuple[tuple[float, ...], ...] | None = None,
) -> dict[str, Any] | None:
    """Return selected-layout feasibility metrics for spatial tasks."""

    if episode.task_family == TASK_NEXT_TO:
        return layout_next_to_feasibility(
            layout,
            object_footprint_half_extents,
            bin_footprint_half_extents,
            episode.objects,
            int(episode.target_object_id),
            int(episode.referent_object_ids[0]),
            table_bounds,
            robot_bounding_box=robot_bounding_box,
        )
    if episode.task_family == TASK_BETWEEN:
        return layout_between_feasibility(
            layout,
            object_footprint_half_extents,
            bin_footprint_half_extents,
            episode.objects,
            int(episode.target_object_id),
            episode.referent_object_ids,
            table_bounds,
            robot_bounding_box=robot_bounding_box,
        )
    if episode.task_family == TASK_MOVE:
        if episode.direction is None:
            raise ValueError("Move feasibility requires an episode direction.")
        return layout_move_feasibility(
            layout,
            object_footprint_half_extents,
            bin_footprint_half_extents,
            episode.objects,
            int(episode.target_object_id),
            str(episode.direction),
            table_bounds,
            robot_bounding_box=robot_bounding_box,
        )
    return None


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


def layout_object_surface_distances(
    layout: dict[str, Any],
    object_footprint_half_extents: dict[str, Any],
    object_names_by_slot: list[str] | tuple[str, ...] | None = None,
) -> list[float]:
    """Compute top-down surface distances between all object footprint pairs."""

    object_vertices = _layout_object_vertices(layout, object_footprint_half_extents, object_names_by_slot)
    distances = []
    for left_index, left_vertices in enumerate(object_vertices):
        for right_vertices in object_vertices[left_index + 1 :]:
            distances.append(polygon_surface_distance(left_vertices, right_vertices))
    return distances


def layout_min_object_surface_distance(
    layout: dict[str, Any],
    object_footprint_half_extents: dict[str, Any],
    object_names_by_slot: list[str] | tuple[str, ...] | None = None,
) -> float | None:
    """Compute the minimum top-down surface distance between object footprints."""

    distances = layout_object_surface_distances(layout, object_footprint_half_extents, object_names_by_slot)
    if not distances:
        return None
    return min(distances)


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
    min_object_surface_distance_m: float,
    table_bounds: TableBounds | None,
    min_table_bounds_inside_fraction: float,
    robot_bounding_box: Polygon2D | None,
    robot_clearance_m: float,
) -> float:
    if table_bounds is not None and not _table_bounds_contain_fraction(
        candidate.vertices,
        table_bounds,
        min_inside_fraction=min_table_bounds_inside_fraction,
    ):
        return -math.inf

    if (
        robot_bounding_box is not None
        and polygon_surface_distance(candidate.vertices, robot_bounding_box) < robot_clearance_m
    ):
        return -math.inf

    bin_distance = polygon_surface_distance(candidate.vertices, bin_vertices)
    if bin_distance < bin_clearance_m:
        return -math.inf

    if not placed:
        return bin_distance

    min_object_distance = math.inf
    for existing in placed:
        surface_distance = polygon_surface_distance(candidate.vertices, existing.vertices)
        if surface_distance <= min_object_surface_distance_m + _EPS:
            return -math.inf
        min_object_distance = min(min_object_distance, surface_distance)
    return min_object_distance + 0.05 * bin_distance


def _sampled_object_footprint_parts(
    object_footprint: tuple[Point2D, Point2D] | tuple[Point2D, Point2D, MoveFootprintBoxes],
) -> tuple[Point2D, Point2D, MoveFootprintBoxes]:
    if len(object_footprint) == 2:
        half_extents, center_offset = object_footprint
        return half_extents, center_offset, ()
    return object_footprint


def _sample_spaced_footprints(
    object_footprints: list[tuple[Point2D, Point2D] | tuple[Point2D, Point2D, MoveFootprintBoxes]],
    object_bin_clearance_margins: list[float],
    polygon: Polygon2D,
    bin_vertices: Polygon2D,
    rng: random.Random,
    bin_clearance_m: float,
    max_attempts: int,
    candidates_per_object: int,
    min_object_surface_distance_m: float,
    layout_constraint: LayoutConstraint | None = None,
    layout_constraint_name: str | None = None,
    candidate_score_adjustment: CandidateScoreAdjustment | None = None,
    candidate_center_suggestions: CandidateCenterSuggestions | None = None,
    preferred_first_object_id: int | None = None,
    preferred_placement_order: list[int] | tuple[int, ...] | None = None,
    stop_after_valid_attempts: int | None = None,
    sample_random_valid_layout: bool = False,
    table_bounds: TableBounds | None = None,
    min_table_bounds_inside_fraction: float = MIN_INITIAL_TABLE_BOUNDS_INSIDE_FRACTION,
    robot_bounding_box: Polygon2D | None = None,
    robot_clearance_m: float = MIN_ROBOT_SURFACE_DISTANCE_M,
) -> tuple[list[_PlacedFootprint], int, int, float, float, dict[str, int]]:
    count = len(object_footprints)
    base_placement_order = sorted(
        range(count),
        key=lambda object_id: math.hypot(*object_footprints[object_id][0]),
        reverse=True,
    )

    top_layouts: list[tuple[float, float, float, list[_PlacedFootprint]]] = []
    best_footprints: list[_PlacedFootprint] | None = None
    best_min_distance = -math.inf
    best_bin_distance = -math.inf
    valid_attempts = 0
    rejection_counts = {
        "object_placement_failed": 0,
        "bin_clearance_failed": 0,
        "object_clearance_failed": 0,
        "task_feasibility_failed": 0,
    }
    attempts_run = 0
    for attempt in range(1, max_attempts + 1):
        attempts_run = attempt
        placement_order = list(base_placement_order)
        rng.shuffle(placement_order)
        if preferred_placement_order is not None:
            prefix = []
            for object_id in preferred_placement_order:
                if object_id in placement_order and object_id not in prefix:
                    prefix.append(object_id)
            placement_order = [*prefix, *(object_id for object_id in placement_order if object_id not in prefix)]
        elif preferred_first_object_id is not None:
            placement_order.remove(preferred_first_object_id)
            placement_order.insert(0, preferred_first_object_id)
        yaws = [rng.uniform(-math.pi, math.pi) for _ in range(count)]
        placed: list[_PlacedFootprint] = []
        failed = False

        for object_id in placement_order:
            best_candidate: _PlacedFootprint | None = None
            best_score = -math.inf
            suggested_centers = (
                candidate_center_suggestions(object_id, yaws[object_id], placed)
                if candidate_center_suggestions is not None
                else []
            )
            candidate_centers = [
                *suggested_centers,
                *(_sample_point_in_polygon(polygon, rng) for _ in range(candidates_per_object)),
            ]
            for center in candidate_centers:
                if not _point_in_polygon(center, polygon):
                    continue
                half_extents, center_offset, move_footprint_boxes = _sampled_object_footprint_parts(
                    object_footprints[object_id]
                )
                candidate = _PlacedFootprint(
                    object_id=object_id,
                    center=center,
                    yaw=yaws[object_id],
                    half_extents=half_extents,
                    center_offset=center_offset,
                    vertices=_footprint_vertices(center, half_extents, center_offset, yaws[object_id]),
                    move_footprint_boxes=move_footprint_boxes,
                )
                score = _score_candidate(
                    candidate,
                    placed,
                    bin_vertices,
                    bin_clearance_m + object_bin_clearance_margins[object_id],
                    min_object_surface_distance_m,
                    table_bounds,
                    min_table_bounds_inside_fraction,
                    robot_bounding_box,
                    robot_clearance_m,
                )
                if math.isfinite(score) and candidate_score_adjustment is not None:
                    score += float(candidate_score_adjustment([*placed, candidate]))
                if score > best_score:
                    best_candidate = candidate
                    best_score = score

            if best_candidate is None or not math.isfinite(best_score):
                failed = True
                break
            placed.append(best_candidate)

        if failed:
            rejection_counts["object_placement_failed"] += 1
            continue

        footprints_by_id = sorted(placed, key=lambda footprint: footprint.object_id)
        if not _bin_clearance_ok(footprints_by_id, bin_vertices, bin_clearance_m, object_bin_clearance_margins):
            rejection_counts["bin_clearance_failed"] += 1
            continue
        min_distance = _minimum_surface_distance(footprints_by_id)
        bin_distance = _minimum_bin_surface_distance(footprints_by_id, bin_vertices)
        if min_distance is None or min_distance <= min_object_surface_distance_m + _EPS:
            rejection_counts["object_clearance_failed"] += 1
            continue
        task_feasibility = layout_constraint(footprints_by_id) if layout_constraint is not None else None
        if layout_constraint is not None and task_feasibility is None:
            rejection_counts["task_feasibility_failed"] += 1
            continue
        layout_selection_score = (
            float(task_feasibility["layout_selection_score"])
            if isinstance(task_feasibility, dict) and "layout_selection_score" in task_feasibility
            else min_distance
        )
        valid_attempts += 1
        top_layouts.append((layout_selection_score, min_distance, bin_distance, footprints_by_id))
        if not sample_random_valid_layout:
            top_layouts.sort(key=lambda entry: (entry[0], entry[1], entry[2]), reverse=True)
            del top_layouts[DEFAULT_TOP_VALID_LAYOUT_CANDIDATES:]
        if min_distance > best_min_distance or (
            math.isclose(min_distance, best_min_distance) and bin_distance > best_bin_distance
        ):
            best_footprints = footprints_by_id
            best_min_distance = min_distance
            best_bin_distance = bin_distance
        if stop_after_valid_attempts is not None and valid_attempts >= stop_after_valid_attempts:
            break

    if count == 4 and top_layouts:
        _selected_score, selected_min_distance, selected_bin_distance, selected_footprints = rng.choice(top_layouts)
        return (
            selected_footprints,
            attempts_run,
            valid_attempts,
            selected_min_distance,
            selected_bin_distance,
            rejection_counts,
        )
    if best_footprints is not None:
        return best_footprints, attempts_run, valid_attempts, best_min_distance, best_bin_distance, rejection_counts

    constraint_clause = f" and satisfying {layout_constraint_name}" if layout_constraint_name else ""
    raise LayoutGenerationError(
        f"Could not sample {count} non-overlapping object placements with "
        f">={bin_clearance_m:.5f} m bin surface clearance{constraint_clause} "
        f"after {max_attempts} attempts. Rejections: {rejection_counts}."
    )


def _single_object_footprint(
    object_footprint: tuple[Point2D, Point2D] | tuple[Point2D, Point2D, MoveFootprintBoxes],
    object_bin_clearance_margin_m: float,
    polygon: Polygon2D,
    bin_vertices: Polygon2D,
    rng: random.Random,
    bin_clearance_m: float,
    max_attempts: int,
    table_bounds: TableBounds | None = None,
    min_table_bounds_inside_fraction: float = MIN_INITIAL_TABLE_BOUNDS_INSIDE_FRACTION,
    robot_bounding_box: Polygon2D | None = None,
    robot_clearance_m: float = MIN_ROBOT_SURFACE_DISTANCE_M,
) -> tuple[_PlacedFootprint, int, int, float]:
    valid_footprints: list[tuple[_PlacedFootprint, float]] = []
    valid_attempts = 0
    for attempt in range(1, max_attempts + 1):
        center = _sample_point_in_polygon(polygon, rng)
        yaw = rng.uniform(-math.pi, math.pi)
        half_extents, center_offset, move_footprint_boxes = _sampled_object_footprint_parts(object_footprint)
        footprint = _PlacedFootprint(
            object_id=0,
            center=center,
            yaw=yaw,
            half_extents=half_extents,
            center_offset=center_offset,
            vertices=_footprint_vertices(center, half_extents, center_offset, yaw),
            move_footprint_boxes=move_footprint_boxes,
        )
        if table_bounds is not None and not _table_bounds_contain_fraction(
            footprint.vertices,
            table_bounds,
            min_inside_fraction=min_table_bounds_inside_fraction,
        ):
            continue
        if (
            robot_bounding_box is not None
            and polygon_surface_distance(footprint.vertices, robot_bounding_box) < robot_clearance_m
        ):
            continue
        bin_distance = polygon_surface_distance(footprint.vertices, bin_vertices)
        if bin_distance < bin_clearance_m + object_bin_clearance_margin_m:
            continue
        valid_attempts += 1
        valid_footprints.append((footprint, bin_distance))

    if valid_footprints:
        selected_footprint, selected_bin_distance = rng.choice(valid_footprints)
        return selected_footprint, max_attempts, valid_attempts, selected_bin_distance

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
    min_object_surface_distance_m: float = MIN_OBJECT_SURFACE_DISTANCE_M,
    table_bounds: TableBounds | None = None,
    next_to_success_distance_m: float = SPATIAL_SUCCESS_DISTANCE_M,
    move_boundary_gap_m: float = MOVE_FEASIBILITY_MIN_BOUNDARY_GAP_M,
    move_straightness_tolerance_m: float = MOVE_STRAIGHTNESS_TOLERANCE_M,
    robot_bounding_box: list[tuple[float, ...]] | tuple[tuple[float, ...], ...] | None = None,
    robot_clearance_m: float = MIN_ROBOT_SURFACE_DISTANCE_M,
    sample_random_valid_spatial_layout: bool = False,
) -> dict[str, Any]:
    """Sample a replayable initial scene layout for one episode."""

    if not bin_random_poses:
        raise ValueError("Expected at least one bin pose.")
    if len(bin_random_poses) != len(valid_spawn_regions):
        raise ValueError(
            f"Expected one valid object spawn region per bin pose, got "
            f"{len(valid_spawn_regions)} regions for {len(bin_random_poses)} bin poses."
        )
    if not episode.objects:
        raise ValueError("Cannot generate a layout for an episode with no objects.")
    if episode.task_family == TASK_MOVE and len(episode.objects) != 4:
        raise ValueError("Move layout generation requires exactly four objects.")
    if episode.task_family in {TASK_NEXT_TO, TASK_BETWEEN, TASK_MOVE} and table_bounds is None:
        raise ValueError(f"{episode.task_family} layout generation requires table_bounds.")

    min_object_surface_distance_m = max(float(min_object_surface_distance_m), 0.0)
    next_to_success_distance_m = max(float(next_to_success_distance_m), 0.0)
    move_boundary_gap_m = max(float(move_boundary_gap_m), 0.0)
    move_straightness_tolerance_m = max(float(move_straightness_tolerance_m), 0.0)
    robot_clearance_m = max(float(robot_clearance_m), 0.0)
    bin_half_extents, bin_center_offset = _coerce_footprint(
        bin_footprint_half_extents,
        DEFAULT_BIN_FOOTPRINT_HALF_EXTENTS,
    )
    raw_object_footprints = [object_footprint_half_extents.get(object_name) for object_name in episode.objects]
    object_footprints = [
        (
            *_coerce_footprint(raw_footprint, DEFAULT_OBJECT_FOOTPRINT_HALF_EXTENTS),
            _object_move_footprint_boxes(object_name, raw_footprint),
        )
        for object_name, raw_footprint in zip(episode.objects, raw_object_footprints, strict=True)
    ]
    object_bin_clearance_margins = [
        _coerce_bin_clearance_margin(raw_footprint) for raw_footprint in raw_object_footprints
    ]
    sample_random_valid_layout = (
        bool(sample_random_valid_spatial_layout)
        and episode.task_family in {TASK_NEXT_TO, TASK_BETWEEN}
    )

    candidate_bin_pose_indices = list(range(len(bin_random_poses))) if episode.task_family == TASK_BIN else [0]
    attempted_bin_pose_indices: list[int] = []
    last_error: LayoutGenerationError | None = None
    while len(attempted_bin_pose_indices) < len(candidate_bin_pose_indices):
        if not attempted_bin_pose_indices:
            bin_pose_index = candidate_bin_pose_indices[rng.randrange(len(candidate_bin_pose_indices))]
        else:
            remaining_indices = [
                index for index in candidate_bin_pose_indices if index not in attempted_bin_pose_indices
            ]
            bin_pose_index = remaining_indices[rng.randrange(len(remaining_indices))]
        attempted_bin_pose_indices.append(bin_pose_index)

        bin_translation, bin_rpy = bin_random_poses[bin_pose_index]
        bin_vertices = _footprint_vertices(
            (float(bin_translation[0]), float(bin_translation[1])),
            bin_half_extents,
            bin_center_offset,
            float(bin_rpy[2]),
        )
        polygon = _xy_polygon(valid_spawn_regions[bin_pose_index])
        layout_constraint: LayoutConstraint | None = None
        layout_constraint_name: str | None = None
        candidate_score_adjustment: CandidateScoreAdjustment | None = None
        candidate_center_suggestions: CandidateCenterSuggestions | None = None
        preferred_first_object_id: int | None = None
        preferred_placement_order: list[int] | None = None
        stop_after_valid_attempts: int | None = None
        robot_polygon = _xy_polygon(robot_bounding_box) if robot_bounding_box is not None else None
        if episode.task_family == TASK_NEXT_TO:
            assert table_bounds is not None
            target_object_id = int(episode.target_object_id)
            referent_object_id = int(episode.referent_object_ids[0])

            def layout_constraint(footprints: list[_PlacedFootprint]) -> dict[str, Any] | None:
                return _next_to_feasibility(
                    footprints,
                    target_object_id,
                    referent_object_id,
                    bin_vertices,
                    table_bounds,
                    next_to_success_distance_m,
                    min_object_surface_distance_m,
                    robot_polygon,
                    robot_clearance_m,
                )

            layout_constraint_name = "next-to feasibility"
        elif episode.task_family == TASK_BETWEEN:
            assert table_bounds is not None
            target_object_id = int(episode.target_object_id)
            referent_object_ids = (
                int(episode.referent_object_ids[0]),
                int(episode.referent_object_ids[1]),
            )

            def layout_constraint(footprints: list[_PlacedFootprint]) -> dict[str, Any] | None:
                return _between_feasibility(
                    footprints,
                    target_object_id,
                    referent_object_ids,
                    bin_vertices,
                    table_bounds,
                    BETWEEN_LINE_TOLERANCE_M,
                    BETWEEN_FEASIBILITY_MIN_SEGMENT_FRACTION,
                    min_object_surface_distance_m,
                    robot_polygon,
                    robot_clearance_m,
                )

            layout_constraint_name = "between feasibility"
        elif episode.task_family == TASK_MOVE:
            assert table_bounds is not None
            target_object_id = int(episode.target_object_id)
            if episode.direction is None:
                raise ValueError("Move layout generation requires an episode direction.")
            direction = str(episode.direction)

            def layout_constraint(footprints: list[_PlacedFootprint]) -> dict[str, Any] | None:
                return _move_feasibility(
                    footprints,
                    target_object_id,
                    direction,
                    bin_vertices,
                    table_bounds,
                    move_boundary_gap_m,
                    move_straightness_tolerance_m,
                    robot_polygon,
                )

            layout_constraint_name = "move clear path"
            preferred_first_object_id = target_object_id
            move_axis, _move_sign = _move_direction_axis_and_sign(direction)
            boundary_object_id = min(
                (object_id for object_id in range(len(object_footprints)) if object_id != target_object_id),
                key=lambda object_id: object_footprints[object_id][0][move_axis],
            )
            preferred_placement_order = [target_object_id, boundary_object_id]

            def candidate_center_suggestions(
                object_id: int,
                yaw: float,
                placed: list[_PlacedFootprint],
            ) -> list[Point2D]:
                if object_id == target_object_id:
                    return _move_target_candidate_centers(
                        object_footprints[object_id],
                        yaw,
                        direction,
                        table_bounds,
                        move_boundary_gap_m,
                    )
                target = next(
                    (footprint for footprint in placed if footprint.object_id == target_object_id),
                    None,
                )
                if target is None:
                    return []
                return _move_boundary_candidate_centers(
                    target,
                    object_footprints[object_id],
                    yaw,
                    direction,
                    move_boundary_gap_m,
                )

            def candidate_score_adjustment(footprints: list[_PlacedFootprint]) -> float:
                return _move_candidate_score_adjustment(
                    footprints,
                    target_object_id,
                    direction,
                    table_bounds,
                    move_boundary_gap_m,
                )

        try:
            if len(episode.objects) == 1:
                footprint, attempts, valid_attempts, min_bin_surface_distance_m = _single_object_footprint(
                    object_footprints[0],
                    object_bin_clearance_margins[0],
                    polygon,
                    bin_vertices,
                    rng,
                    bin_clearance_m,
                    max_attempts,
                    table_bounds=table_bounds,
                    robot_bounding_box=robot_polygon,
                    robot_clearance_m=robot_clearance_m,
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
                    rejection_counts,
                ) = _sample_spaced_footprints(
                    object_footprints,
                    object_bin_clearance_margins,
                    polygon,
                    bin_vertices,
                    rng,
                    bin_clearance_m,
                    max_attempts,
                    candidates_per_object,
                    min_object_surface_distance_m,
                    layout_constraint,
                    layout_constraint_name,
                    candidate_score_adjustment,
                    candidate_center_suggestions,
                    preferred_first_object_id,
                    preferred_placement_order,
                    stop_after_valid_attempts,
                    sample_random_valid_layout=sample_random_valid_layout,
                    table_bounds=table_bounds,
                    robot_bounding_box=robot_polygon
                    if episode.task_family in {TASK_NEXT_TO, TASK_BETWEEN}
                    else None,
                    robot_clearance_m=robot_clearance_m,
                )
            if len(episode.objects) == 1:
                rejection_counts = {}

            task_feasibility = layout_constraint(footprints) if layout_constraint is not None else None
            if layout_constraint is not None and task_feasibility is None:
                raise LayoutGenerationError(
                    f"Selected layout unexpectedly failed {layout_constraint_name or 'task feasibility'}."
                )
        except LayoutGenerationError as exc:
            last_error = exc
            continue

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
                "rejection_counts": rejection_counts,
                "min_between_object_surface_distance_m": min_between_object_surface_distance_m,
                "min_surface_distance_m": min_between_object_surface_distance_m,
                "min_bin_surface_distance_m": min_bin_surface_distance_m,
                "required_min_bin_surface_distance_m": bin_clearance_m,
                "required_min_between_object_surface_distance_m": min_object_surface_distance_m,
                "required_min_move_boundary_gap_m": (
                    move_boundary_gap_m if episode.task_family == TASK_MOVE else None
                ),
                "required_min_move_clear_path_m": (
                    max(move_boundary_gap_m - 1.0e-6, 0.0) if episode.task_family == TASK_MOVE else None
                ),
                "required_min_robot_surface_distance_m": (
                    robot_clearance_m
                    if robot_bounding_box is not None and episode.task_family in {TASK_NEXT_TO, TASK_BETWEEN}
                    else None
                ),
                "move_straightness_tolerance_m": (
                    move_straightness_tolerance_m if episode.task_family == TASK_MOVE else None
                ),
                "required_bin_surface_distance_m_by_object": [
                    bin_clearance_m + margin for margin in object_bin_clearance_margins
                ],
                "valid_spawn_region_index": bin_pose_index,
                "bin_pose_attempts": len(attempted_bin_pose_indices),
                "layout_selection": "random_valid" if sample_random_valid_layout else "top_valid",
                "task_feasibility": task_feasibility,
                "max_attempts": max_attempts,
                "candidates_per_object": candidates_per_object if len(episode.objects) > 1 else None,
            },
        }

    raise LayoutGenerationError(
        f"Could not sample a non-overlapping layout for episode {episode_index} across "
        f"{len(candidate_bin_pose_indices)} bin pose(s). These objects may be too large. Last error: {last_error}"
    )
