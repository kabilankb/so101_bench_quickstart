"""Isaac Lab ManagerBased environment configs for SO-101 Bench."""

from __future__ import annotations

import math
import os

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, TiledCameraCfg
from isaaclab.utils import configclass
from isaacsim.core.utils.rotations import euler_angles_to_quat

from so101_bench import assets, mdp
from so101_bench.assets.so101 import SO101_CONTACT_GRASP_CFG
from so101_bench.benchmark import TASK_BETWEEN, TASK_BIN, TASK_MIXED, TASK_MOVE, TASK_NEXT_TO

ASSETS_PATH = os.path.dirname(os.path.abspath(assets.__file__))

UI_CPU_PHYSICS_ENV_VAR = "SO101_BENCH_UI_CPU_PHYSICS"

TABLE_TOP_Z = 1.02
INCH_TO_M = 0.0254
BEDROOM_TABLETOP_USD = f"{ASSETS_PATH}/usd/room_scan.usdc"
BEDROOM_TABLETOP_COLLISION_PRIM = "collision_plane/collision_plane_001/collision_plane"
TABLETOP_SHORT_SIDE_IN = 20.0

OBJECT_ASSET_NAMES = ["object_1", "object_2", "object_3", "object_4"]
OBJECT_LABELS = ["white pen", "black pen", "blue pen", "red pen"]
TABLE_BOUNDS = {"x": (-10.0, 10.0), "y": (-10.0, 10.0)}
BIN_FIXED_TRANSLATION = (-0.04424, -0.69022, 1.04161)
BIN_FIXED_YAW_DEG = -79.392
BIN_ROOT_ROTATION_RPY_DEG = (90.0, 0.0, 0.0)
BIN_FIXED_POSE = (BIN_FIXED_TRANSLATION[0], BIN_FIXED_TRANSLATION[1], math.radians(BIN_FIXED_YAW_DEG))
OBJECT_FIXED_POSES = (
    (-0.22312, -0.80442, math.radians(90.0)),
    (-0.47272, -0.70795, math.radians(90.0)),
    (-0.36156, -0.82500, math.radians(90.0)),
    (-0.49656, -0.82500, math.radians(90.0)),
)
BIN_X_RANGE = (-0.34, -0.26)
BIN_Y_RANGE = (-0.72, -0.54)
MIN_RESET_TIME_S = 0.5
MIN_FAILURE_TIME_S = 1.0
SUCCESS_CONFIRM_TIME_S = 0.25
PHYSICS_DT = 1.0 / 120.0
CONTROL_DT = 1.0 / 30.0
CONTROL_DECIMATION = int(round(CONTROL_DT / PHYSICS_DT))
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_FOCAL_LENGTH = 13.5
DEFAULT_CAMERA_HORIZONTAL_APERTURE = 20.955
IPHONE_17_PRO_MAIN_CAMERA_HORIZONTAL_FOV_DEG = 74.0
IPHONE_17_PRO_MAIN_CAMERA_HORIZONTAL_APERTURE = 34.61329224445431
IPHONE_17_PRO_MAIN_CAMERA_FOCAL_LENGTH = IPHONE_17_PRO_MAIN_CAMERA_HORIZONTAL_APERTURE / (
    2.0 * math.tan(math.radians(IPHONE_17_PRO_MAIN_CAMERA_HORIZONTAL_FOV_DEG) / 2.0)
)

INNOMAKER_WRIST_CAMERA_WIDTH = DEFAULT_CAMERA_WIDTH
INNOMAKER_WRIST_CAMERA_HEIGHT = DEFAULT_CAMERA_HEIGHT
INNOMAKER_WRIST_CAMERA_UPDATE_PERIOD = CONTROL_DT
INNOMAKER_WRIST_CAMERA_HORIZONTAL_FOV_DEG = 45.0
INNOMAKER_WRIST_CAMERA_HORIZONTAL_APERTURE = 20.955
INNOMAKER_WRIST_CAMERA_FOCAL_LENGTH = INNOMAKER_WRIST_CAMERA_HORIZONTAL_APERTURE / (
    2.0 * math.tan(math.radians(INNOMAKER_WRIST_CAMERA_HORIZONTAL_FOV_DEG) / 2.0)
)
INNOMAKER_WRIST_CAMERA_CLIPPING_RANGE = (0.02, 3.0)
INNOMAKER_WRIST_CAMERA_FOCUS_DISTANCE = 0.5
INNOMAKER_WRIST_CAMERA_F_STOP = 0.0
INNOMAKER_WRIST_CAMERA_POS = (-0.005, 0.060, -0.062)
INNOMAKER_WRIST_CAMERA_RPY_DEG = (-45.0, 0.0, 0.0)
BEDROOM_LIGHT_TEMPERATURE_K = 6000.0
BEDROOM_LIGHT_INTENSITY = 12000.0
BEDROOM_LIGHT_RADIUS = 0.4


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    unique_points = sorted(set(map(tuple, points)))
    if len(unique_points) <= 1:
        return np.array(unique_points, dtype=float)

    def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)

    return np.array(lower[:-1] + upper[:-1], dtype=float)


def _minimum_area_rectangle_size(points: np.ndarray) -> tuple[float, float]:
    hull = _convex_hull_2d(points)
    if len(hull) == 0:
        return 0.0, 0.0
    if len(hull) == 1:
        return 0.0, 0.0
    if len(hull) == 2:
        return float(np.linalg.norm(hull[1] - hull[0])), 0.0

    best_area = math.inf
    best_size = (0.0, 0.0)
    for index in range(len(hull)):
        edge = hull[(index + 1) % len(hull)] - hull[index]
        edge_length = np.linalg.norm(edge)
        if edge_length == 0.0:
            continue

        axis_x = edge / edge_length
        axis_y = np.array([-axis_x[1], axis_x[0]])
        width = float(np.ptp(hull @ axis_x))
        height = float(np.ptp(hull @ axis_y))
        area = width * height
        if area < best_area:
            best_area = area
            best_size = (width, height)

    return best_size


def _collision_prim_size(usd_path: str, collision_subpath: str) -> tuple[float, float, float] | None:
    if not usd_path or not os.path.exists(usd_path) or not collision_subpath:
        return None

    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        return None

    normalized_subpath = collision_subpath.strip("/")
    prim = stage.GetPrimAtPath("/" + normalized_subpath)
    if not prim.IsValid():
        root = stage.GetDefaultPrim()
        if root and root.IsValid():
            prim = stage.GetPrimAtPath(root.GetPath().AppendPath(normalized_subpath))

    if not prim.IsValid():
        leaf_name = normalized_subpath.split("/")[-1]
        matches = [candidate for candidate in stage.Traverse() if candidate.GetName() == leaf_name]
        if not matches:
            return None
        prim = matches[0]

    if not prim.IsA(UsdGeom.Mesh):
        mesh_matches = [
            candidate
            for candidate in Usd.PrimRange(prim)
            if candidate.IsA(UsdGeom.Mesh) and candidate.GetName() == prim.GetName()
        ]
        if not mesh_matches:
            mesh_matches = [candidate for candidate in Usd.PrimRange(prim) if candidate.IsA(UsdGeom.Mesh)]
        if not mesh_matches:
            return None
        prim = mesh_matches[0]

    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    if not points:
        return None

    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    world_points = np.array(
        [transform.Transform(Gf.Vec3d(point[0], point[1], point[2])) for point in points],
        dtype=float,
    )
    centered_points = world_points - world_points.mean(axis=0)

    _, _, axes = np.linalg.svd(centered_points, full_matrices=False)
    plane_points = centered_points @ axes[:2].T
    normal_coordinates = centered_points @ axes[2]

    side_a, side_b = _minimum_area_rectangle_size(plane_points)
    thickness = float(np.ptp(normal_coordinates))
    return float(side_a), float(side_b), thickness


def _bedroom_tabletop_scale() -> tuple[float, float, float]:
    robot_base_extent = 0.087  # the sim extent matching 2.875 real inches (base vertical extent)
    target_short_side = robot_base_extent * (20.0 / 2.875)

    size = _collision_prim_size(BEDROOM_TABLETOP_USD, BEDROOM_TABLETOP_COLLISION_PRIM)
    # print("COLLISION MESH SIZE:", size)
    raw_short_side = min(size[0:2])
    scale = target_short_side / raw_short_side
    # print("BEDROOM MESH SCALE:",(scale, scale, scale))
    return (scale, scale, scale)

BEDROOM_TABLETOP_SCALE = _bedroom_tabletop_scale()


def _validate_fixed_object_poses() -> None:
    x_min, x_max = TABLE_BOUNDS["x"]
    y_min, y_max = TABLE_BOUNDS["y"]
    for index, (x, y, _yaw) in enumerate(OBJECT_FIXED_POSES, start=1):
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            raise ValueError(
                f"OBJECT_FIXED_POSES[{index - 1}] for object_{index} is outside TABLE_BOUNDS: "
                f"(x={x}, y={y}), bounds x={TABLE_BOUNDS['x']} y={TABLE_BOUNDS['y']}."
            )


_validate_fixed_object_poses()


def _robot_cfg() -> ArticulationCfg:
    cfg = SO101_CONTACT_GRASP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    return cfg


def _camera_cfg(
    width: int = DEFAULT_CAMERA_WIDTH,
    height: int = DEFAULT_CAMERA_HEIGHT,
    update_period: float = 0.0,
    focal_length: float = DEFAULT_CAMERA_FOCAL_LENGTH,
    horizontal_aperture: float = DEFAULT_CAMERA_HORIZONTAL_APERTURE,
    vertical_aperture: float | None = None,
    clipping_range: tuple[float, float] = (0.01, 5.0),
    focus_distance: float = 0.35,
    f_stop: float = 0.0,
) -> TiledCameraCfg:
    return TiledCameraCfg(
        prim_path="",
        update_period=update_period,
        height=height,
        width=width,
        data_types=["rgb"],
        # colorize_instance_segmentation=True,
        spawn=sim_utils.PinholeCameraCfg(
            projection_type="pinhole",
            focal_length=focal_length,
            focus_distance=focus_distance,
            f_stop=f_stop,
            horizontal_aperture=horizontal_aperture,
            vertical_aperture=vertical_aperture,
            clipping_range=clipping_range,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=euler_angles_to_quat(np.array([0.0, 0.0, 0.0]), degrees=True),
            convention="opengl",
        ),
    )


def _object_spawn(
    color: tuple[float, float, float],
    size: tuple[float, float, float],
    mass: float = 0.035,
) -> sim_utils.CuboidCfg:
    return sim_utils.CuboidCfg(
        size=size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_depenetration_velocity=1.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.7),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )


@configclass
class So101BenchSceneCfg(InteractiveSceneCfg):
    """Bedroom tabletop scene with SO-101, bin, cameras, and four object slots."""

    env_spacing = 1.5
    num_envs = 1

    robot: ArticulationCfg = _robot_cfg()

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper",
                name="gripper",
            ),
        ],
    )

    bedroom_tabletop = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BedroomTabletop",
        spawn=sim_utils.UsdFileCfg(
            usd_path=BEDROOM_TABLETOP_USD,
            scale=BEDROOM_TABLETOP_SCALE,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=euler_angles_to_quat(np.array([180.0, 0.0, 0.0]), degrees=True),
        ),
    )

    plastic_bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PlasticBin",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSETS_PATH}/usd/plastic_bin.usdc",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=BIN_FIXED_TRANSLATION,
            rot=euler_angles_to_quat(
                np.array(
                    [
                        BIN_ROOT_ROTATION_RPY_DEG[0],
                        BIN_ROOT_ROTATION_RPY_DEG[1],
                        BIN_ROOT_ROTATION_RPY_DEG[2] + BIN_FIXED_YAW_DEG,
                    ]
                ),
                degrees=True,
            ),
        ),
    )

    object_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object_1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSETS_PATH}/usd/objects/blue_bowl.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
                max_depenetration_velocity=0.5,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            )
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(OBJECT_FIXED_POSES[0][0], OBJECT_FIXED_POSES[0][1], 1.04168)
        ),
    )

    object_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object_2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSETS_PATH}/usd/objects/yellow_screwdriver.usdc",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
                max_depenetration_velocity=0.5,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(OBJECT_FIXED_POSES[1][0], OBJECT_FIXED_POSES[1][1], 1.04294)
        ),
    )

    object_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object_3",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSETS_PATH}/usd/objects/silver_glasses.usdc",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
                max_depenetration_velocity=0.5,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(OBJECT_FIXED_POSES[2][0], OBJECT_FIXED_POSES[2][1], 1.04294)
        ),
    )

    object_4 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object_4",
        spawn=_object_spawn((0.90, 0.16, 0.10), (0.095, 0.014, 0.014), mass=0.020),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(OBJECT_FIXED_POSES[3][0], OBJECT_FIXED_POSES[3][1], 1.07)
        ),
    )

    camera_wrist = _camera_cfg(
        width=INNOMAKER_WRIST_CAMERA_WIDTH,
        height=INNOMAKER_WRIST_CAMERA_HEIGHT,
        update_period=INNOMAKER_WRIST_CAMERA_UPDATE_PERIOD,
        focal_length=INNOMAKER_WRIST_CAMERA_FOCAL_LENGTH,
        horizontal_aperture=INNOMAKER_WRIST_CAMERA_HORIZONTAL_APERTURE,
        vertical_aperture=None,
        clipping_range=INNOMAKER_WRIST_CAMERA_CLIPPING_RANGE,
        focus_distance=INNOMAKER_WRIST_CAMERA_FOCUS_DISTANCE,
        f_stop=INNOMAKER_WRIST_CAMERA_F_STOP,
    )
    camera_wrist.prim_path = "{ENV_REGEX_NS}/Robot/gripper/gripper_cam"
    camera_wrist.offset.pos = INNOMAKER_WRIST_CAMERA_POS
    camera_wrist.offset.rot = euler_angles_to_quat(np.array(INNOMAKER_WRIST_CAMERA_RPY_DEG), degrees=True)

    camera_overhead = _camera_cfg(
        update_period=CONTROL_DT,
        focal_length=IPHONE_17_PRO_MAIN_CAMERA_FOCAL_LENGTH,
        horizontal_aperture=IPHONE_17_PRO_MAIN_CAMERA_HORIZONTAL_APERTURE,
        vertical_aperture=None,
    )
    camera_overhead.prim_path = "{ENV_REGEX_NS}/CameraOverhead"
    camera_overhead.offset.pos = (-0.11342, -0.41946, 1.58316)
    camera_overhead.offset.rot = (0.176819, 0.084379, 0.417046, 0.887518)
    camera_overhead.offset.convention = "opengl"

    # bedroom_rear_light = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/BedroomRearLight",
    #     spawn=sim_utils.SphereLightCfg(
    #         intensity=BEDROOM_LIGHT_INTENSITY,
    #         color=(1.0, 0.92, 0.78),
    #         enable_color_temperature=True,
    #         color_temperature=BEDROOM_LIGHT_TEMPERATURE_K,
    #         radius=BEDROOM_LIGHT_RADIUS,
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.44865, 0.47924, 3.09735)),
    # )
    # bedroom_front_light = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/BedroomFrontLight",
    #     spawn=sim_utils.SphereLightCfg(
    #         intensity=10000.0,
    #         color=(1.0, 0.92, 0.78),
    #         enable_color_temperature=True,
    #         color_temperature=5500,
    #         radius=0.5,
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -1.70292, 2.9382)),
    # )

    bedroom_domelight = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BedroomDomeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=800,
            color=(1.0, 0.92, 0.78),
            enable_color_temperature=True,
            color_temperature=7500,
        ),
    )


@configclass
class ActionsCfg:
    """Absolute joint-position actions matching the SO-101 workshop and GR00T runner."""

    joint_positions = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"],
        scale=1.0,
        use_default_offset=False,
    )


@configclass
class ObservationsCfg:
    """Robot proprioception plus wrist and overhead camera observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_obs = ObsTerm(func=mdp.joint_pos)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        ee_frame_state = ObsTerm(
            func=mdp.ee_frame_state,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class VisualCfg(ObsGroup):
        rgb_wrist = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("camera_wrist"), "data_type": "rgb", "normalize": False},
        )
        rgb_overhead = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("camera_overhead"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    visual: VisualCfg = VisualCfg()


def _reset_scene_event(task_family: str, object_count_range: tuple[int, int]) -> EventTerm:
    return EventTerm(
        func=mdp.reset_benchmark_scene,
        mode="reset",
        params={
            "object_asset_names": OBJECT_ASSET_NAMES,
            "bin_name": "plastic_bin",
            "object_labels": OBJECT_LABELS,
            "task_family": task_family,
            "object_count_range": object_count_range,
            "table_bounds": TABLE_BOUNDS,
            "table_top_z": TABLE_TOP_Z,
            "min_object_spacing": 0.105,
            "bin_fixed_pose": BIN_FIXED_POSE,
            "bin_root_rotation": tuple(math.radians(angle) for angle in BIN_ROOT_ROTATION_RPY_DEG),
            "bin_z": BIN_FIXED_TRANSLATION[2],
            "object_fixed_poses": OBJECT_FIXED_POSES,
            "randomize_bin_for_bin_task": False,
            "bin_x_range": BIN_X_RANGE,
            "bin_y_range": BIN_Y_RANGE,
        },
    )


@configclass
class EventCfg:
    """Reset events for robot appearance, robot pose, and benchmark layout."""

    reset_robot_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"],
            ),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_set_robot_visual_material = EventTerm(
        func=mdp.randomize_robot_color,
        mode="reset",
        params={"color_names": ["orange"]},
    )

    reset_benchmark_scene = _reset_scene_event(TASK_MIXED, (1, 4))


@configclass
class TerminationsCfg:
    """Paper-derived success and measurable failure terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    success = DoneTerm(
        func=mdp.task_success,
        time_out=False,
        params={
            "object_asset_names": OBJECT_ASSET_NAMES,
            "bin_name": "plastic_bin",
            "table_bounds": TABLE_BOUNDS,
            "min_episode_time_s": MIN_RESET_TIME_S,
            "confirm_time_s": SUCCESS_CONFIRM_TIME_S,
        },
    )

    failure = DoneTerm(
        func=mdp.benchmark_failure,
        time_out=False,
        params={
            "object_asset_names": OBJECT_ASSET_NAMES,
            "bin_name": "plastic_bin",
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "min_episode_time_s": MIN_FAILURE_TIME_S,
            "displacement_baseline_time_s": MIN_FAILURE_TIME_S,
        },
    )


@configclass
class So101BenchEnvCfg(ManagerBasedRLEnvCfg):
    """Base SO-101 Bench environment suitable for teleop, GR00T rollout, or eval."""

    scene: So101BenchSceneCfg = So101BenchSceneCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    rewards = None
    commands = None
    curriculum = None

    def __post_init__(self) -> None:
        self.decimation = CONTROL_DECIMATION
        self.episode_length_s = 20.0
        self.scene.num_envs = 1
        self.sim.dt = PHYSICS_DT
        self.sim.render_interval = self.decimation
        self.sim.render.rendering_mode = "quality"
        if os.environ.get(UI_CPU_PHYSICS_ENV_VAR, "").lower() in {"1", "true", "yes", "on"}:
            self.sim.device = "cpu"
            self.sim.use_fabric = False
        self.viewer.eye = (0.03, -0.62, 0.48)
        self.viewer.lookat = (0.24, 0.0, 0.08)


@configclass
class So101BenchBinEnvCfg(So101BenchEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.events.reset_benchmark_scene = _reset_scene_event(TASK_BIN, (1, 4))


@configclass
class So101BenchNextToEnvCfg(So101BenchEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.events.reset_benchmark_scene = _reset_scene_event(TASK_NEXT_TO, (2, 4))


@configclass
class So101BenchBetweenEnvCfg(So101BenchEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.events.reset_benchmark_scene = _reset_scene_event(TASK_BETWEEN, (3, 4))


@configclass
class So101BenchMoveEnvCfg(So101BenchEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.events.reset_benchmark_scene = _reset_scene_event(TASK_MOVE, (1, 4))
