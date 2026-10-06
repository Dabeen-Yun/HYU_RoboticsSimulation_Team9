# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ant with ground-property estimation: teacher-student (RMA-style) + foot contact sensors.

Hypothesis: the policy loses performance on unseen ground because it cannot tell the ground properties (friction,
contact model: analytic ground plane vs triangle mesh, payload). If those properties are known (teacher) or can be
inferred from recent contact history (student), one policy can use the right gait on each ground.

Observation groups (same terms and order, the teacher only appends privileged information):
    policy  (student, deployable): original 60 | height scan 273 | foot contacts 16 | 8-step history
    teacher (privileged):          policy obs | effective friction, mass ratio, analytic-plane flag

Reward terms are inherited unchanged. Evaluation environments use the original terminations.
"""

import torch

import isaaclab.terrains as terrain_gen

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_assets.robots.ant import ANT_CFG  # isort: skip

from .ant_env_cfg import EventCfg, TerminationsCfg
from .ant_rough_env_cfg import (
    ANT_TEST_TERRAINS_CFG,
    ANT_TRAIN_TERRAINS_CFG,
    AntRoughEnvCfg,
    AntRoughEventCfg,
    AntRoughObservationsCfg,
    AntRoughSceneCfg,
    AntTestEventCfg,
    _original_plane_cfg,
    _rehearsal_mask,
    _terrain_cfg,
)

FEET = ["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]
HISTORY = 8

##
# Observation terms
##


def foot_contact_features(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    force_scale: float = 0.1,
    contact_threshold: float = 0.5,
) -> torch.Tensor:
    """Per foot: normal force, tangential force, tangential/normal ratio (friction usage), slip speed in contact."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    normal = forces[..., 2].clamp(min=0.0)
    tangential = torch.norm(forces[..., :2], dim=-1)
    ratio = tangential / (normal + 1.0)
    in_contact = (normal > contact_threshold).float()
    foot_speed = torch.norm(env.scene[asset_cfg.name].data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    slip = foot_speed * in_contact
    return torch.cat([normal * force_scale, tangential * force_scale, ratio, slip], dim=-1)


_COMBINE_PRIORITY = {"average": 0, "min": 1, "multiply": 2, "max": 3}


def _combine(a: torch.Tensor, b: float, mode: str) -> torch.Tensor:
    if mode == "average":
        return 0.5 * (a + b)
    if mode == "min":
        return torch.clamp(a, max=b)
    if mode == "multiply":
        return a * b
    return torch.clamp(a, min=b)


def privileged_physics(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Ground truth that the deployed robot cannot measure directly (teacher only):
    effective foot-ground friction - 1, total mass ratio - 1, 1 if the robot spawned on the analytic ground plane."""
    # the observation manager calls this term once at construction (before the startup randomization events), so the
    # value is only cached once the simulation is stepping
    if "_privileged_physics" not in env.__dict__ or not env.__dict__["_privileged_physics_final"]:
        asset = env.scene[asset_cfg.name]
        view = asset.root_physx_view
        robot_friction = view.get_material_properties()[..., 0].mean(dim=-1).to(env.device)
        ground = env.cfg.scene.terrain.physics_material
        mode = max(["average", ground.friction_combine_mode], key=lambda m: _COMBINE_PRIORITY[m])
        friction = _combine(robot_friction, ground.static_friction, mode)
        mass_ratio = (view.get_masses().sum(-1) / asset.data.default_mass.sum(-1)).to(env.device)
        if env.cfg.scene.terrain.terrain_type == "plane":
            plane = torch.ones(env.num_envs, device=env.device)
        else:
            plane = _rehearsal_mask(env, ("flat",)).float()
        env.__dict__["_privileged_physics"] = torch.stack([friction - 1.0, mass_ratio - 1.0, plane], dim=-1)
        env.__dict__["_privileged_physics_final"] = env.common_step_counter > 0
    return env.__dict__["_privileged_physics"]


def randomize_friction_per_env(
    env: ManagerBasedRLEnv,
    env_ids,
    friction_range: tuple[float, float],
    num_buckets: int = 64,
    log_uniform: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Startup event: one friction value per robot (all shapes), i.e. one ground material per env.

    The default material randomization samples every collision shape independently, so the per-robot average is
    almost constant and the "ground friction" is not identifiable. Values are quantized to ``num_buckets`` levels
    (PhysX limits the number of unique materials)."""
    view = env.scene[asset_cfg.name].root_physx_view
    materials = view.get_material_properties()
    ids = torch.arange(env.num_envs) if env_ids is None else env_ids.cpu()
    if log_uniform:
        levels = torch.logspace(torch.log10(torch.tensor(friction_range[0])), torch.log10(torch.tensor(friction_range[1])), num_buckets)
    else:
        levels = torch.linspace(friction_range[0], friction_range[1], num_buckets)
    f = levels[torch.randint(0, num_buckets, (len(ids),))]
    materials[ids, :, 0] = f[:, None]
    materials[ids, :, 1] = f[:, None]
    materials[ids, :, 2] = 0.0
    view.set_material_properties(materials, ids)


@configclass
class AntRMAEventCfg(AntRoughEventCfg):
    """Training events of Isaac-Ant-RMA-v0: as Isaac-Ant-Rough-v0, but friction is sampled per robot (same range)."""

    physics_material = EventTerm(func=randomize_friction_per_env, mode="startup", params={"friction_range": (0.4, 1.2)})


_CONTACT_PARAMS = {
    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET, preserve_order=True),
    "asset_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True),
}

##
# Scene
##


@configclass
class AntRMASceneCfg(AntRoughSceneCfg):
    robot = ANT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot", spawn=ANT_CFG.spawn.replace(activate_contact_sensors=True)
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_foot", history_length=0, track_air_time=False)


##
# Observations
##


@configclass
class AntRMAObservationsCfg(AntRoughObservationsCfg):
    @configclass
    class PolicyCfg(AntRoughObservationsCfg.PolicyCfg):
        """Student: original 60 | height scan 273 | contacts 16 | history 8 x (8 + 8 + 3 + 3 + 16)."""

        foot_contacts = ObsTerm(func=foot_contact_features, params=_CONTACT_PARAMS, noise=Unoise(n_min=-0.02, n_max=0.02))
        hist_joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2, history_length=HISTORY)
        hist_actions = ObsTerm(func=mdp.last_action, history_length=HISTORY)
        hist_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=HISTORY)
        hist_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=HISTORY)
        hist_contacts = ObsTerm(func=foot_contact_features, params=_CONTACT_PARAMS, history_length=HISTORY)

    @configclass
    class TeacherCfg(PolicyCfg):
        """Teacher: student observations + privileged ground/robot properties."""

        privileged = ObsTerm(func=privileged_physics)

    policy: PolicyCfg = PolicyCfg()
    teacher: TeacherCfg = TeacherCfg()


##
# Environments
##


@configclass
class AntRMAEnvCfg(AntRoughEnvCfg):
    """Training environment (teacher PPO and student distillation): Isaac-Ant-RMA-v0."""

    scene: AntRMASceneCfg = AntRMASceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)
    observations: AntRMAObservationsCfg = AntRMAObservationsCfg()
    events: AntRMAEventCfg = AntRMAEventCfg()


def _to_eval_rma(cfg: AntRMAEnvCfg, terrain):
    cfg.scene.terrain = terrain
    cfg.scene.ground_plane = None
    for group in (cfg.observations.policy, cfg.observations.teacher):
        group.height_scan.params["plane_height"] = None
        group.enable_corruption = False
    cfg.terminations = TerminationsCfg()


@configclass
class AntRMATestEnvCfg(AntRMAEnvCfg):
    """Held-out environment (same as Isaac-Ant-Rough-Test-v0): Isaac-Ant-RMA-Test-v0."""

    events: AntTestEventCfg = AntTestEventCfg()

    def __post_init__(self):
        super().__post_init__()
        _to_eval_rma(self, _terrain_cfg(ANT_TEST_TERRAINS_CFG))


@configclass
class AntRMAFlatEnvCfg(AntRMAEnvCfg):
    """Original Isaac-Ant-v0 conditions (same as Isaac-Ant-Rough-Flat-v0): Isaac-Ant-RMA-Flat-v0."""

    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        _to_eval_rma(self, _original_plane_cfg())


##
# Wide randomization (Isaac-Ant-RMA-Wide-*): train on a much broader range of grounds and robot conditions
##


def randomize_motor_strength(
    env: ManagerBasedRLEnv, env_ids, strength_range: tuple[float, float], action_name: str = "joint_effort"
):
    """Startup event: per-robot actuator strength (torque scale). The reward is computed on the raw actions and is
    therefore unchanged."""
    term = env.action_manager.get_term(action_name)
    strength = torch.empty(env.num_envs, device=env.device).uniform_(*strength_range)
    base = term._scale if isinstance(term._scale, float) else term._scale[0, 0].item()
    term._scale = base * strength[:, None].expand(-1, term.action_dim).clone()
    env.__dict__["_motor_strength"] = strength


def motor_strength_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Privileged (teacher only): actuator strength - 1."""
    strength = env.__dict__.get("_motor_strength")
    if strength is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return (strength - 1.0).unsqueeze(-1)


ANT_TRAIN_WIDE_TERRAINS_CFG = ANT_TRAIN_TERRAINS_CFG.replace(
    sub_terrains={
        **ANT_TRAIN_TERRAINS_CFG.sub_terrains,
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.02, 0.14),
            step_width=0.4,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.02, 0.12), platform_width=2.0
        ),
    }
)
"""Wider training terrain. The test terrain types (wave, discrete obstacles, rough) are not widened."""


@configclass
class AntRMAWideSceneCfg(AntRMASceneCfg):
    terrain = _terrain_cfg(ANT_TRAIN_WIDE_TERRAINS_CFG)


@configclass
class AntRMAWideEventCfg(AntRMAEventCfg):
    torso_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso"),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)},
        },
    )
    motor_strength = EventTerm(func=randomize_motor_strength, mode="startup", params={"strength_range": (0.7, 1.3)})

    def __post_init__(self):
        super().__post_init__()
        self.physics_material.params = {"friction_range": (0.1, 2.0), "log_uniform": True}
        self.body_mass.params["mass_distribution_params"] = (0.6, 1.5)
        self.push_robot.interval_range_s = (3.0, 6.0)
        self.push_robot.params["velocity_range"] = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}


@configclass
class AntRMAWideObservationsCfg(AntRMAObservationsCfg):
    @configclass
    class TeacherCfg(AntRMAObservationsCfg.TeacherCfg):
        motor_strength = ObsTerm(func=motor_strength_obs)

    teacher: TeacherCfg = TeacherCfg()


@configclass
class AntRMAWideEnvCfg(AntRMAEnvCfg):
    """Wide-randomization training environment: Isaac-Ant-RMA-Wide-v0.
    friction 0.1-2.0 (log-uniform, per robot), mass x0.6-1.5, torso CoM +-3 cm, motor strength x0.7-1.3,
    pushes +-1 m/s every 3-6 s, higher stairs/boxes/slopes."""

    scene: AntRMAWideSceneCfg = AntRMAWideSceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)
    observations: AntRMAWideObservationsCfg = AntRMAWideObservationsCfg()
    events: AntRMAWideEventCfg = AntRMAWideEventCfg()


@configclass
class AntRMAWideTestEnvCfg(AntRMATestEnvCfg):
    """Held-out environment with the wide teacher observations: Isaac-Ant-RMA-Wide-Test-v0."""

    observations: AntRMAWideObservationsCfg = AntRMAWideObservationsCfg()


@configclass
class AntRMAWideFlatEnvCfg(AntRMAFlatEnvCfg):
    """Original environment with the wide teacher observations: Isaac-Ant-RMA-Wide-Flat-v0."""

    observations: AntRMAWideObservationsCfg = AntRMAWideObservationsCfg()

