# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ant environments for generalization to unseen terrains.

The reward terms are inherited unchanged from :class:`AntEnvCfg` (they are the evaluation metric), and all evaluation
environments also use the original terminations. Only the scene (terrain, sensors), observations, events (domain
randomization) and, for training only, the torso-height termination are modified.

Layout of the generated terrain:
    * rows (x-axis): difficulty increases with the row index. All robots spawn in row 0 and walk towards +x
      (the progress target is at (1000, 0, 0)), so walking further means facing harder terrain.
    * cols (y-axis): terrain types, allocated by their ``proportion``.
    The baseline Ant covers roughly 140 m per 16 s episode, so the grid is 20 rows x 8 m = 160 m long.

Contact model:
    The original Isaac-Ant-v0 ground is a USD ground plane (analytic plane collider). Triangle-mesh terrain has a
    noticeably different contact behaviour: the original policy drops from ~130 to ~50 reward on a perfectly flat
    *mesh*. The training scene therefore keeps the original ground plane at z = 0: flat columns are sunk below it so
    the robots walk on the real plane there, and all other terrain is raised on top of it.
"""

import math

import numpy as np
import torch
import trimesh

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCaster, RayCasterCfg, patterns
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp

from .ant_env_cfg import AntEnvCfg, EventCfg, MySceneCfg, ObservationsCfg, TerminationsCfg

##
# Terrains
##


def sunken_flat_terrain(difficulty: float, cfg: "SunkenFlatTerrainCfg") -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """A thin slab below z = 0. Used together with a ground plane at z = 0, so that the robot walks on the plane."""
    box = trimesh.creation.box(
        extents=(cfg.size[0], cfg.size[1], 0.1),
        transform=trimesh.transformations.translation_matrix((cfg.size[0] / 2, cfg.size[1] / 2, cfg.depth - 0.05)),
    )
    return [box], np.array([cfg.size[0] / 2, cfg.size[1] / 2, 0.0])


@configclass
class SunkenFlatTerrainCfg(SubTerrainBaseCfg):
    function = sunken_flat_terrain
    depth: float = -0.5
    """Top of the slab (m). Must be below the ground plane."""


ANT_TRAIN_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=20,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        # walked on the original ground plane (see module docstring)
        "flat": SunkenFlatTerrainCfg(proportion=0.3),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(0.0, 0.06), noise_step=0.01, border_width=0.25
        ),
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.02, 0.10),
            step_width=0.4,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.02, 0.08), platform_width=2.0
        ),
    },
)
"""Training terrains. All terrain stays at or above z = 0 (the ground plane)."""

ANT_TEST_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=24,
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=20,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        # unseen terrain type
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.4, amplitude_range=(0.02, 0.08), num_waves=4, border_width=0.25
        ),
        # unseen terrain type
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.3,
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.3, 1.0),
            obstacle_height_range=(0.02, 0.10),
            num_obstacles=40,
            platform_width=2.0,
            border_width=0.25,
        ),
        # seen terrain type, but rougher than in training (noise max 0.06 -> 0.10)
        "random_rough_hard": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3, noise_range=(0.0, 0.10), noise_step=0.01, border_width=0.25
        ),
    },
)
"""Held-out terrains. Never used for training. Pure triangle mesh (no ground plane). The generator seed is fixed for
reproducible evaluation."""

ANT_FLAT_MESH_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=24,
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=20,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=(0.0, 0.0), noise_step=0.005, border_width=0.0
        )
    },
)
"""Perfectly flat triangle-mesh ground (control for the contact model)."""


def _terrain_cfg(generator: TerrainGeneratorCfg) -> TerrainImporterCfg:
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=generator,
        # all robots start in row 0 and progress towards harder rows by walking
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=_ground_material(),
        debug_vis=False,
    )


def _ground_material() -> sim_utils.RigidBodyMaterialCfg:
    # "multiply" so that the robot-side friction randomization sets the effective friction directly
    return sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    )


def _original_plane_cfg() -> TerrainImporterCfg:
    """Exactly the original Isaac-Ant-v0 ground."""
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )


##
# Height-scan helpers
##


def _ground_heights(sensor: RayCaster, plane_height: float | None = None) -> torch.Tensor:
    """Terrain height under each ray. If the scene also contains a ground plane at ``plane_height`` (training scene),
    the scanned mesh is combined with it (the scanner itself only sees the terrain mesh)."""
    hits_z = sensor.data.ray_hits_w[..., 2]
    if plane_height is not None:
        hits_z = torch.where(torch.isfinite(hits_z), hits_z.clamp(min=plane_height), torch.full_like(hits_z, plane_height))
    return hits_z


def height_scan_with_plane(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, plane_height: float | None = None, offset: float = 0.5
) -> torch.Tensor:
    """Same as :func:`isaaclab.envs.mdp.height_scan` (torso height - ground height - offset), plane-aware."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    return sensor.data.pos_w[:, 2].unsqueeze(1) - _ground_heights(sensor, plane_height) - offset


def root_height_above_terrain_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    sensor_cfg: SceneEntityCfg,
    plane_height: float | None = None,
    radius: float = 0.3,
) -> torch.Tensor:
    """Terminate when the torso is less than ``minimum_height`` above the terrain directly under it.

    On flat ground (z = 0) this is identical to the original ``root_height_below_minimum`` term. On raised terrain
    the original (world-frame) term is too permissive: the policy learned a crouched gait on stairs/slopes that
    fell below 0.31 m as soon as it reached flat ground.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    hits_z = _ground_heights(sensor, plane_height)
    dist_xy = torch.norm(sensor.data.ray_hits_w[..., :2] - sensor.data.pos_w[:, None, :2], dim=-1)
    under = (dist_xy < radius) & torch.isfinite(hits_z)
    ground_z = torch.where(under, hits_z, torch.zeros_like(hits_z)).sum(-1) / under.sum(-1).clamp(min=1)
    return (sensor.data.pos_w[:, 2] - ground_z) < minimum_height


@configclass
class AntRoughTerminationsCfg(TerminationsCfg):
    """Training only: original terminations, with the torso height measured relative to the terrain."""

    torso_height = DoneTerm(
        func=root_height_above_terrain_below_minimum,
        params={"minimum_height": 0.31, "sensor_cfg": SceneEntityCfg("height_scanner"), "plane_height": 0.0},
    )


##
# Scene
##


@configclass
class AntRoughSceneCfg(MySceneCfg):
    """Training scene: original ground plane + generated terrain + height scanner."""

    terrain = _terrain_cfg(ANT_TRAIN_TERRAINS_CFG)

    ground_plane = AssetBaseCfg(
        prim_path="/World/groundPlane",
        spawn=sim_utils.GroundPlaneCfg(physics_material=_ground_material()),
    )

    # height map around the torso. Aligned with the world frame because the walking direction is always +x.
    # 21 x 13 = 273 rays covering x in [-0.5, 1.5] m (look-ahead) and y in [-0.6, 0.6] m.
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.5, 0.0, 20.0)),
        ray_alignment="world",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[2.0, 1.2]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


##
# MDP settings
##


@configclass
class AntRoughObservationsCfg(ObservationsCfg):
    """Original Ant observations + height scan (torso height relative to the terrain under each ray)."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        height_scan = ObsTerm(
            func=height_scan_with_plane,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "plane_height": 0.0},
            noise=Unoise(n_min=-0.02, n_max=0.02),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class AntRoughEventCfg(EventCfg):
    """Original reset events + domain randomization."""

    # -- startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.4, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    # -- interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 8.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

    def __post_init__(self):
        # random initial heading (original: none)
        self.reset_base.params["pose_range"] = {"yaw": (-math.pi / 6, math.pi / 6)}


def _apply_test_events(events: EventCfg):
    """Held-out physics: lower friction and wider mass range than training, no pushes, original reset."""
    if hasattr(events, "physics_material"):
        events.physics_material.params["static_friction_range"] = (0.3, 0.7)
        events.physics_material.params["dynamic_friction_range"] = (0.3, 0.7)
    if hasattr(events, "body_mass"):
        events.body_mass.params["mass_distribution_params"] = (0.7, 1.3)
    if hasattr(events, "push_robot"):
        events.push_robot = None
    events.reset_base.params["pose_range"] = {}


##
# Rehearsal: the flat (ground plane) columns keep the exact original Isaac-Ant-v0 conditions
##


def _rehearsal_mask(env: ManagerBasedRLEnv, terrain_names: tuple[str, ...]) -> torch.Tensor:
    """Boolean mask of the envs that spawn on a sub-terrain column whose name is in ``terrain_names``."""
    key = ("_rehearsal_mask", terrain_names)
    if key not in env.__dict__:
        terrain = env.scene.terrain
        gen = env.cfg.scene.terrain.terrain_generator
        mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if gen is not None and getattr(terrain, "terrain_types", None) is not None:
            names = list(gen.sub_terrains.keys())
            props = np.array([c.proportion for c in gen.sub_terrains.values()])
            props = np.cumsum(props / props.sum())
            col_is = [names[int(np.min(np.where(i / gen.num_cols + 0.001 < props)[0]))] in terrain_names for i in range(gen.num_cols)]
            mask = torch.tensor(col_is, device=env.device)[terrain.terrain_types]
        env.__dict__[key] = mask
    return env.__dict__[key]


def restore_nominal_physics(
    env: ManagerBasedRLEnv, env_ids, terrain_names: tuple[str, ...], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """Startup event (after the randomization terms): undo friction/mass randomization for the rehearsal envs."""
    ids = torch.nonzero(_rehearsal_mask(env, terrain_names)).flatten().cpu()
    if len(ids) == 0:
        return
    asset = env.scene[asset_cfg.name]
    view = asset.root_physx_view
    materials = view.get_material_properties()
    materials[ids, :, 0] = 1.0
    materials[ids, :, 1] = 1.0
    materials[ids, :, 2] = 0.0
    view.set_material_properties(materials, ids)
    masses = view.get_masses()
    masses[ids] = asset.data.default_mass[ids].to(masses.device)
    view.set_masses(masses, ids)
    inertias = view.get_inertias()
    inertias[ids] = asset.data.default_inertia[ids].to(inertias.device)
    view.set_inertias(inertias, ids)


def push_except_rehearsal(env: ManagerBasedRLEnv, env_ids: torch.Tensor, velocity_range: dict, terrain_names: tuple[str, ...]):
    env_ids = env_ids[~_rehearsal_mask(env, terrain_names)[env_ids]]
    if len(env_ids) > 0:
        mdp.push_by_setting_velocity(env, env_ids, velocity_range)


def reset_root_state_rehearsal(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor, pose_range: dict, velocity_range: dict, terrain_names: tuple[str, ...]
):
    """Random initial heading, except for the rehearsal envs (original reset)."""
    mask = _rehearsal_mask(env, terrain_names)[env_ids]
    if mask.any():
        mdp.reset_root_state_uniform(env, env_ids[mask], {}, velocity_range)
    if (~mask).any():
        mdp.reset_root_state_uniform(env, env_ids[~mask], pose_range, velocity_range)


@configclass
class AntRoughRehearsalEventCfg(AntRoughEventCfg):
    """Training events: domain randomization on the rough columns, original conditions on the flat columns."""

    restore_nominal = EventTerm(func=restore_nominal_physics, mode="startup", params={"terrain_names": ("flat",)})

    def __post_init__(self):
        super().__post_init__()
        self.push_robot.func = push_except_rehearsal
        self.push_robot.params["terrain_names"] = ("flat",)
        self.reset_base.func = reset_root_state_rehearsal
        self.reset_base.params["terrain_names"] = ("flat",)


@configclass
class AntTestEventCfg(AntRoughEventCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_test_events(self)


##
# Environments (height scan)
##


@configclass
class AntRoughEnvCfg(AntEnvCfg):
    """Training environment: Isaac-Ant-Rough-v0."""

    # clone_in_fabric must be off: the ray caster looks up the torso prims of every env on the USD stage
    scene: AntRoughSceneCfg = AntRoughSceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)
    observations: AntRoughObservationsCfg = AntRoughObservationsCfg()
    events: AntRoughEventCfg = AntRoughEventCfg()
    terminations: AntRoughTerminationsCfg = AntRoughTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # update the height scanner once per policy step
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt


@configclass
class AntRoughRehearsalEnvCfg(AntRoughEnvCfg):
    """Variant (experiment v5): the flat columns keep the exact original conditions (no randomization, no pushes,
    no random heading) to rehearse the original task. Isaac-Ant-Rough-Rehearsal-v0."""

    events: AntRoughRehearsalEventCfg = AntRoughRehearsalEventCfg()


def _to_eval(cfg: AntRoughEnvCfg, terrain: TerrainImporterCfg):
    """Turn the training config into an evaluation config: given ground only, original terminations, clean sensors."""
    cfg.scene.terrain = terrain
    cfg.scene.ground_plane = None
    cfg.observations.policy.height_scan.params["plane_height"] = None
    cfg.observations.policy.enable_corruption = False
    cfg.terminations = TerminationsCfg()


@configclass
class AntRoughTestEnvCfg(AntRoughEnvCfg):
    """Held-out environment for height-scan policies: Isaac-Ant-Rough-Test-v0."""

    events: AntTestEventCfg = AntTestEventCfg()

    def __post_init__(self):
        super().__post_init__()
        _to_eval(self, _terrain_cfg(ANT_TEST_TERRAINS_CFG))


@configclass
class AntRoughFlatEnvCfg(AntRoughEnvCfg):
    """Original Isaac-Ant-v0 conditions for height-scan policies: Isaac-Ant-Rough-Flat-v0.

    Original ground plane, nominal physics (no randomization, no pushes, original reset), original terminations,
    clean sensors. Only difference to Isaac-Ant-v0: the height scan observation (the ray caster handles USD planes).
    """

    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        _to_eval(self, _original_plane_cfg())


##
# Environments without height scan (baseline / ablation / controls)
##


@configclass
class AntRoughBlindEnvCfg(AntEnvCfg):
    """Ablation: same training scene, randomization and termination as Isaac-Ant-Rough-v0, but no height scan in the
    observations (the scanner is only used by the termination term). Isaac-Ant-Rough-Blind-v0."""

    scene: AntRoughSceneCfg = AntRoughSceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)
    events: AntRoughEventCfg = AntRoughEventCfg()
    terminations: AntRoughTerminationsCfg = AntRoughTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt


@configclass
class AntTestBlindEnvCfg(AntEnvCfg):
    """Same held-out environment as Isaac-Ant-Rough-Test-v0, with the original observations: Isaac-Ant-Test-Blind-v0.

    Used to evaluate blind policies (the original Isaac-Ant-v0 policy, the ablation) on the same terrain and physics.
    """

    events: AntTestEventCfg = AntTestEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain = _terrain_cfg(ANT_TEST_TERRAINS_CFG)


@configclass
class AntBlindFlatEnvCfg(AntEnvCfg):
    """Control: original Isaac-Ant-v0, but the ground is a perfectly flat triangle mesh instead of the USD ground
    plane. Isolates the effect of the contact model of mesh terrain. Isaac-Ant-Blind-Flat-v0."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain = _terrain_cfg(ANT_FLAT_MESH_TERRAINS_CFG)
