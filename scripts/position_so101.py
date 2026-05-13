"""
Isaac Sim Script Editor script for positioning the SO-101 relative to the collision plane's (tabletop's) edges

Used these measurements:
- Back right corner of base is 9.5 inches away from short table edge
- Back left corner of base is 0.625 inches away from long edge (his left butt)
- Back right corner is 1 inch away from long edge

Misc measurements:
- Robot base is 2.875 inches tall (Z axis), extent 0.875
- Tabletop is 17 inches tall
- Tabletop is 20 inches by 40 inches
- Cards against humanity is 7.5 inches


back_left_to_long_side: 0.021013 sim units = 0.694 real inches
back_right_to_long_side: 0.029157 sim units = 0.964 real inches
back_right_to_short_side: 0.287984 sim units = 9.517 real inches
base/table z gap: 0.000144 sim units
"""

import os
import numpy as np
import omni.usd
from pxr import Gf, Usd, UsdGeom

stage = omni.usd.get_context().get_stage()

BASE_FOOTPRINT = "/World/envs/env_0/Robot/base/visuals/base_so101_v2"
BASE_MESH = BASE_FOOTPRINT + "/mesh"
PLANE = "/World/envs/env_0/BedroomTabletop/collision_plane/collision_plane_001/collision_plane"
TABLE_SHORT_SIDE_IN = float(os.getenv("SO101_BENCH_TABLETOP_SHORT_SIDE_IN", "20.0"))


def mesh_world_points(path):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim: {path}")
    mesh = UsdGeom.Mesh(prim)
    xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(
        [xf.Transform(Gf.Vec3d(p[0], p[1], p[2])) for p in mesh.GetPointsAttr().Get()],
        dtype=float,
    )


def robot_back_corners_xy():
    frame_prim = stage.GetPrimAtPath(BASE_FOOTPRINT)
    mesh_prim = stage.GetPrimAtPath(BASE_MESH)

    frame_to_world = UsdGeom.Xformable(frame_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    mesh_to_world = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    world_to_frame = frame_to_world.GetInverse()

    mesh = UsdGeom.Mesh(mesh_prim)
    pts_frame = np.array(
        [
            world_to_frame.Transform(
                mesh_to_world.Transform(Gf.Vec3d(p[0], p[1], p[2]))
            )
            for p in mesh.GetPointsAttr().Get()
        ],
        dtype=float,
    )

    z_min = pts_frame[:, 2].min()
    bottom_pts = pts_frame[pts_frame[:, 2] <= z_min + 1e-4]

    back_left_local = bottom_pts[np.argmax(bottom_pts[:, 0])]
    back_right_local = bottom_pts[np.argmin(bottom_pts[:, 0])]

    back_left_world = np.array(
        frame_to_world.Transform(Gf.Vec3d(*back_left_local)), dtype=float
    )
    back_right_world = np.array(
        frame_to_world.Transform(Gf.Vec3d(*back_right_local)), dtype=float
    )

    return back_left_world[:2], back_right_world[:2]


def tabletop_rect_xy():
    xy = mesh_world_points(PLANE)[:, :2]
    center = xy.mean(axis=0)
    _, _, vh = np.linalg.svd(xy - center, full_matrices=False)

    u, v = vh[0], vh[1]
    span_u = np.ptp((xy - center) @ u)
    span_v = np.ptp((xy - center) @ v)

    if span_v > span_u:
        u, v = v, u
        span_u, span_v = span_v, span_u

    return center, u, v, span_u / 2.0, span_v / 2.0


def point_segment_distance(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    return np.linalg.norm(p - (a + t * ab))


bl, br = robot_back_corners_xy()
center, long_axis, short_axis, half_long, half_short = tabletop_rect_xy()

long_side_sign = 1.0 if np.dot(((bl + br) * 0.5) - center, short_axis) >= 0.0 else -1.0
short_side_sign = 1.0 if np.dot(br - center, long_axis) >= 0.0 else -1.0

long_a = center + long_side_sign * half_short * short_axis - half_long * long_axis
long_b = center + long_side_sign * half_short * short_axis + half_long * long_axis
short_a = center + short_side_sign * half_long * long_axis - half_short * short_axis
short_b = center + short_side_sign * half_long * long_axis + half_short * short_axis

measurements = {
    "back_left_to_long_side": point_segment_distance(bl, long_a, long_b),
    "back_right_to_long_side": point_segment_distance(br, long_a, long_b),
    "back_right_to_short_side": point_segment_distance(br, short_a, short_b),
}

sim_units_per_real_inch = (2.0 * half_short) / TABLE_SHORT_SIDE_IN

for name, value in measurements.items():
    print(
        f"{name}: {value:.6f} sim units = {value / sim_units_per_real_inch:.3f} real inches"
    )

table_z = np.median(mesh_world_points(PLANE)[:, 2])
base_bottom_z = mesh_world_points(BASE_MESH)[:, 2].min()
print(f"base/table z gap: {base_bottom_z - table_z:.6f} sim units")
