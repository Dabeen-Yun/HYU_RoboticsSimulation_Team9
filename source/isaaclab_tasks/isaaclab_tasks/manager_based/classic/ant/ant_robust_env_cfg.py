# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""[ant_robust_bundle] Merged unchanged from ant_robust_bundle.zip (teammate experiment, 2026-10-06).

Robustness variants of Isaac-Ant-v0 (assignment 1: an Ant that walks well in unseen environments).

Training variants (all keep the 60-D observation and 8-D action of Isaac-Ant-v0, so every checkpoint
can be evaluated in the original task and in any unseen environment):

    DR                      domain randomization (friction, mass, CoM, initial heading/velocity)
    DR + Rough              + uneven terrain
    DR + Rough + PushNoise  + random pushes and observation noise
    DR + Rough + Smooth     + penalties on body shaking (roll/pitch rate, vertical velocity, action jerk)
    DR + Rough + PushNoise + Smooth

Evaluation-only "unseen" variants change one physical condition at a time (plus one combined case),
with values outside or at the edge of the training randomization ranges.
"""

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp

from .ant_env_cfg import AntEnvCfg

##
# Terrains
##

# The Ant runs ~140 m in one 16 s episode, so tiles are long along +x (the target direction):
# 5 rows x 40 m = 200 m, and every robot spawns in the first row (max_init_terrain_level=0).
# Heights stay within a few centimeters of z=0 because the observation and the fall check use
# the absolute base height (base_pos_z, root_height_below_minimum).
ROUGH_TRAIN_TERRAIN = TerrainGeneratorCfg(
    seed=0,
    size=(40.0, 8.0),
    border_width=20.0,
    num_rows=5,
    num_cols=6,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    curriculum=False,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.4, noise_range=(0.0, 0.05), noise_step=0.01, border_width=0.25
        ),
        "waves": terrain_gen.HfWaveTerrainCfg(proportion=0.2, amplitude_range=(0.0, 0.05), num_waves=8),
        "bumps": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.2,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.3, 1.0),
            obstacle_height_range=(0.01, 0.04),
            num_obstacles=60,
            platform_width=1.0,
        ),
    },
)

# Unseen test terrain: different layout (seed) and taller features than anything seen in training.
ROUGH_TEST_TERRAIN = ROUGH_TRAIN_TERRAIN.replace(
    seed=12345,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.35, noise_range=(0.0, 0.08), noise_step=0.01, border_width=0.25
        ),
        "waves": terrain_gen.HfWaveTerrainCfg(proportion=0.3, amplitude_range=(0.04, 0.08), num_waves=10),
        "bumps": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.35,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.3, 1.0),
            obstacle_height_range=(0.04, 0.08),
            num_obstacles=100,
            platform_width=1.0,
        ),
    },
)


def _rough_terrain(generator: TerrainGeneratorCfg) -> TerrainImporterCfg:
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=generator,
        max_init_terrain_level=0,
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


def _set_ground_friction(cfg: AntEnvCfg, static: float, dynamic: float):
    """Ground friction for the whole scene. The "min" combine mode overrides the robot's default "average"
    (PhysX picks the higher-priority mode), so the contact friction is min(robot, ground)."""
    for material in (cfg.scene.terrain.physics_material, cfg.sim.physics_material):
        material.static_friction = static
        material.dynamic_friction = dynamic
        material.friction_combine_mode = "min"


##
# Training building blocks
##

FEET = ["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]


def _add_domain_randomization(cfg: AntEnvCfg):
    # friction of every robot body; with a "min" ground the contact friction equals the robot's value
    _set_ground_friction(cfg, 1.0, 1.0)
    cfg.events.robot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.25, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    cfg.events.torso_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso"),
            "mass_distribution_params": (0.7, 1.5),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    cfg.events.torso_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.02, 0.02)},
        },
    )
    # initial heading and velocity (the default always starts facing the target, at rest)
    cfg.events.reset_base.params = {
        "pose_range": {"yaw": (-math.pi, math.pi)},
        "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.5, 0.5)},
    }


def _add_rough_terrain(cfg: AntEnvCfg):
    friction = cfg.scene.terrain.physics_material
    cfg.scene.terrain = _rough_terrain(ROUGH_TRAIN_TERRAIN)
    cfg.scene.terrain.physics_material = friction


def _add_push_and_noise(cfg: AntEnvCfg):
    cfg.events.push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )
    # sensor noise; added before each term's scale, so values are in raw units
    obs = cfg.observations.policy
    obs.enable_corruption = True
    obs.base_height.noise = Unoise(n_min=-0.01, n_max=0.01)
    obs.base_lin_vel.noise = Unoise(n_min=-0.1, n_max=0.1)
    obs.base_ang_vel.noise = Unoise(n_min=-0.2, n_max=0.2)
    obs.base_yaw_roll.noise = Unoise(n_min=-0.05, n_max=0.05)
    obs.base_angle_to_target.noise = Unoise(n_min=-0.05, n_max=0.05)
    obs.base_up_proj.noise = Unoise(n_min=-0.02, n_max=0.02)
    obs.base_heading_proj.noise = Unoise(n_min=-0.02, n_max=0.02)
    obs.joint_pos_norm.noise = Unoise(n_min=-0.02, n_max=0.02)
    obs.joint_vel_rel.noise = Unoise(n_min=-0.5, n_max=0.5)


# Each penalty costs about 3% of the baseline policy's episode reward (~131): measured unweighted episode
# sums on the seed-42 baseline were ang_vel_xy 18.4, lin_vel_z 5.2, action_rate 39.7
# (experiments/ant_robust/measure_penalties.py).
SMOOTH_WEIGHTS = {"ang_vel_xy_l2": -0.2, "lin_vel_z_l2": -0.75, "action_rate_l2": -0.1}


def _add_smoothness_penalty(cfg: AntEnvCfg):
    cfg.rewards.ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=SMOOTH_WEIGHTS["ang_vel_xy_l2"])
    cfg.rewards.lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=SMOOTH_WEIGHTS["lin_vel_z_l2"])
    cfg.rewards.action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=SMOOTH_WEIGHTS["action_rate_l2"])


##
# Training variants
##


@configclass
class AntDREnvCfg(AntEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _add_domain_randomization(self)


@configclass
class AntDRRoughEnvCfg(AntDREnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _add_rough_terrain(self)


@configclass
class AntDRRoughPushNoiseEnvCfg(AntDRRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _add_push_and_noise(self)


@configclass
class AntDRRoughSmoothEnvCfg(AntDRRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _add_smoothness_penalty(self)


@configclass
class AntDRRoughPushNoiseSmoothEnvCfg(AntDRRoughPushNoiseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _add_smoothness_penalty(self)


##
# Unseen evaluation variants (never used for training)
##


@configclass
class AntTestLowFrictionEnvCfg(AntEnvCfg):
    """Slippery floor: friction 0.25 / 0.2 (training randomization goes down to 0.3 / 0.25)."""

    def __post_init__(self):
        super().__post_init__()
        _set_ground_friction(self, 0.25, 0.2)


@configclass
class AntTestHeavyEnvCfg(AntEnvCfg):
    """Torso 1.8x heavier (training randomization goes up to 1.5x)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.torso_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso"),
                "mass_distribution_params": (1.8, 1.8),
                "operation": "scale",
                "recompute_inertia": True,
            },
        )


@configclass
class AntTestWeakMotorEnvCfg(AntEnvCfg):
    """Motors deliver 2/3 of the nominal torque (never randomized in training)."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_effort.scale = 5.0


@configclass
class AntTestRoughEnvCfg(AntEnvCfg):
    """Unseen terrain: different layout and up to 8 cm features (training terrain goes up to 5 cm)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain = _rough_terrain(ROUGH_TEST_TERRAIN)


@configclass
class AntTestPushEnvCfg(AntEnvCfg):
    """Strong pushes every 2-4 s (training pushes are up to 1 m/s every 3-6 s)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(2.0, 4.0),
            params={"velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}},
        )


@configclass
class AntTestHeadingEnvCfg(AntEnvCfg):
    """Starts facing a random direction instead of the target."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_base.params = {"pose_range": {"yaw": (-math.pi, math.pi)}, "velocity_range": {}}


@configclass
class AntTestCombinedEnvCfg(AntEnvCfg):
    """Several moderate changes at once: friction 0.5, torso 1.4x, motors 0.85x, unseen terrain,
    pushes, and a random initial heading within +-90 deg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain = _rough_terrain(ROUGH_TEST_TERRAIN)
        _set_ground_friction(self, 0.5, 0.45)
        self.actions.joint_effort.scale = 7.5 * 0.85
        self.events.torso_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso"),
                "mass_distribution_params": (1.4, 1.4),
                "operation": "scale",
                "recompute_inertia": True,
            },
        )
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(3.0, 5.0),
            params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
        )
        self.events.reset_base.params = {"pose_range": {"yaw": (-math.pi / 2, math.pi / 2)}, "velocity_range": {}}


##
# Teammate-terrain variants ("TM"): reconstructed from the team slides (07-08), not from the teammate's code.
#   training : 8 m x 8 m tiles, 20 difficulty rows along +x (harder the farther the Ant walks), 10 columns:
#              flat 3, rough (<= 6 cm) 2, slope 1, stairs (<= 10 cm) 2, boxes (<= 8 cm) 2
#   unseen   : same layout, terrain seed 24, waves 4, obstacles 3, rough (<= 10 cm) 3 columns,
#              friction 0.3-0.7 per contact body, mass x0.7-1.3, no pushes
# During training only, the fall check uses the height above the terrain (as the teammate did against
# "crouching" on raised tiles); evaluation keeps the original absolute check.
##

from isaaclab.sensors import RayCasterCfg, patterns  # noqa: E402

TM_TRAIN_TERRAIN = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=20,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.3),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(proportion=0.2, noise_range=(0.0, 0.06), noise_step=0.01),
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(proportion=0.1, slope_range=(0.0, 0.21), platform_width=2.0),
        "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2, step_height_range=(0.0, 0.10), step_width=0.75, platform_width=2.0
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.0, 0.08), platform_width=2.0
        ),
    },
)

TM_TEST_TERRAIN = TM_TRAIN_TERRAIN.replace(
    seed=24,
    sub_terrains={
        "waves": terrain_gen.HfWaveTerrainCfg(proportion=0.4, amplitude_range=(0.02, 0.08), num_waves=4),
        "obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.3,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.25, 0.75),
            obstacle_height_range=(0.02, 0.10),
            num_obstacles=40,
            platform_width=2.0,
        ),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(proportion=0.3, noise_range=(0.0, 0.10), noise_step=0.01),
    },
)


def root_height_above_terrain_below(env, minimum_height: float, sensor_cfg: SceneEntityCfg,
                                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Fall check relative to the ground right under the torso (single downward ray)."""
    import torch

    asset = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    ground = torch.nan_to_num(sensor.data.ray_hits_w[:, 0, 2], nan=0.0, posinf=0.0, neginf=0.0)
    return asset.data.root_pos_w[:, 2] - ground < minimum_height


def _add_teammate_terrain(cfg: AntEnvCfg):
    friction = cfg.scene.terrain.physics_material
    cfg.scene.terrain = _rough_terrain(TM_TRAIN_TERRAIN)
    cfg.scene.terrain.physics_material = friction
    # the ray caster finds the torsos through USD, so the envs must be cloned into USD (not only into Fabric)
    cfg.scene.clone_in_fabric = False
    cfg.scene.height_probe = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.0, 0.0)),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )
    cfg.terminations.torso_height = DoneTerm(
        func=root_height_above_terrain_below,
        params={"minimum_height": 0.31, "sensor_cfg": SceneEntityCfg("height_probe")},
    )


@configclass
class AntDRTMEnvCfg(AntDREnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _add_teammate_terrain(self)


@configclass
class AntDRTMPushNoiseEnvCfg(AntDRTMEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _add_push_and_noise(self)


@configclass
class AntDRTMSmoothEnvCfg(AntDRTMEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _add_smoothness_penalty(self)


@configclass
class AntDRTMPushNoiseSmoothEnvCfg(AntDRTMPushNoiseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _add_smoothness_penalty(self)


@configclass
class AntTestTMEnvCfg(AntEnvCfg):
    """Reconstruction of the teammate's unseen environment (evaluation only, original fall check)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain = _rough_terrain(TM_TEST_TERRAIN)
        _set_ground_friction(self, 1.0, 1.0)
        self.events.robot_friction = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.3, 0.7),
                "dynamic_friction_range": (0.3, 0.7),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
                "make_consistent": True,
            },
        )
        self.events.robot_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (0.7, 1.3),
                "operation": "scale",
                "recompute_inertia": True,
            },
        )
