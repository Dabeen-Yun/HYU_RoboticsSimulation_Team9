# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A NEW evaluation environment, defined once and registered for both policies.

Edit only the "EDIT HERE" block below (terrain, ground friction, torso mass, pushes). The same environment is then
available under two task IDs:

    Isaac-Ant-New-v0       original 60-D observations -> lecture baseline   (checkpoints/ant_baseline.pt)
    Isaac-Ant-RMA-New-v0   same env + our sensors     -> our final policy    (checkpoints/ant_final.pt)

Both use the original Isaac-Ant-v0 rewards and terminations, so the two scores are directly comparable.
The example below (inverted stairs and slopes, rough patches, friction 0.5, torso 1.3x) is not used anywhere in training.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp

from .ant_env_cfg import AntEnvCfg
from .ant_rma_bundle_eval_cfg import with_rma_sensors
from .ant_robust_env_cfg import _rough_terrain, _set_ground_friction

# ======================================================================== EDIT HERE
NEW_TERRAIN: TerrainGeneratorCfg | None = TerrainGeneratorCfg(
    seed=24,  # fixed so that every run sees the same terrain
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=20,  # difficulty increases with distance (all robots start in row 0)
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "inverted_stairs": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
            proportion=0.4, step_height_range=(0.02, 0.08), step_width=0.5, platform_width=2.0, border_width=0.25
        ),
        "inverted_slope": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.3, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        ),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3, noise_range=(0.0, 0.08), noise_step=0.01, border_width=0.25
        ),
    },
)
"""Terrain generator, or ``None`` for the original infinite flat ground plane."""

GROUND_FRICTION: float | None = 0.5
"""Static = dynamic friction of every robot-ground contact, or ``None`` for the default (1.0)."""

TORSO_MASS_SCALE: float | None = 1.3
"""Torso mass multiplier, or ``None`` for the nominal mass."""

PUSH: tuple[tuple[float, float], float] | None = None
"""((min_interval_s, max_interval_s), max_velocity_m_s), e.g. ((2.0, 4.0), 1.0), or ``None`` for no pushes."""
# ======================================================================== END EDIT


@configclass
class AntNewEnvCfg(AntEnvCfg):
    """Isaac-Ant-v0 with the settings above (60-D observations)."""

    def __post_init__(self):
        super().__post_init__()
        if NEW_TERRAIN is not None:
            self.scene.terrain = _rough_terrain(NEW_TERRAIN)
        if GROUND_FRICTION is not None:
            _set_ground_friction(self, GROUND_FRICTION, GROUND_FRICTION)
        if TORSO_MASS_SCALE is not None:
            self.events.torso_mass = EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="torso"),
                    "mass_distribution_params": (TORSO_MASS_SCALE, TORSO_MASS_SCALE),
                    "operation": "scale",
                    "recompute_inertia": True,
                },
            )
        if PUSH is not None:
            (t0, t1), v = PUSH
            self.events.push_robot = EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(t0, t1),
                params={"velocity_range": {"x": (-v, v), "y": (-v, v)}},
            )


AntRMANewEnvCfg = with_rma_sensors(AntNewEnvCfg, "AntRMANewEnvCfg")
"""Same environment + the height scanner, foot contact sensors and observation history our policy needs."""
