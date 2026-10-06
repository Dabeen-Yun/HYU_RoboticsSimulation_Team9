# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""[ant_robust_bundle] Evaluation of our sensor policy (teacher-student, Isaac-Ant-RMA-*) in the teammate's test envs.

Every class inherits one evaluation environment of ant_robust_env_cfg.py unchanged (terrain, friction, mass, motor,
pushes, initial heading, terminations, rewards) and only adds what our policy observes: the height scanner, the foot
contact sensors and the Isaac-Ant-RMA observation groups (no observation noise, as in the bundle's test envs).

    Isaac-Ant-Test-<X>-v0          bundle env, 60-D observations  (baseline and bundle checkpoints)
    Isaac-Ant-RMA-Bundle-<X>-v0    same env + our sensors          (our policy)
"""

from isaaclab.utils import configclass

from . import ant_robust_env_cfg as bundle
from .ant_rma_env_cfg import AntRMAObservationsCfg, AntRMASceneCfg
from .ant_rough_env_cfg import _original_plane_cfg


@configclass
class AntRMABundleSceneCfg(AntRMASceneCfg):
    """Our sensors on the original ground plane (the bundle env replaces the terrain where it uses one)."""

    terrain = _original_plane_cfg()
    ground_plane = None


def _with_our_sensors(base: type, name: str) -> type:
    @configclass
    class _Cfg(base):
        scene: AntRMABundleSceneCfg = AntRMABundleSceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)
        observations: AntRMAObservationsCfg = AntRMAObservationsCfg()

        def __post_init__(self):
            super().__post_init__()
            for group in (self.observations.policy, self.observations.teacher):
                group.height_scan.params["plane_height"] = None
                group.enable_corruption = False
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt

    _Cfg.__name__ = _Cfg.__qualname__ = name
    return _Cfg


with_rma_sensors = _with_our_sensors
"""Public name: ``with_rma_sensors(AnyAntEnvCfg, "Name")`` adds our sensors to any Isaac-Ant-v0-based environment.
The base environment must set its terrain in ``__post_init__`` (the wrapper's scene starts from the flat plane)."""


BUNDLE_TEST_ENVS = {
    "LowFriction": bundle.AntTestLowFrictionEnvCfg,
    "Heavy": bundle.AntTestHeavyEnvCfg,
    "WeakMotor": bundle.AntTestWeakMotorEnvCfg,
    "Rough": bundle.AntTestRoughEnvCfg,
    "Push": bundle.AntTestPushEnvCfg,
    "Heading": bundle.AntTestHeadingEnvCfg,
    "Combined": bundle.AntTestCombinedEnvCfg,
    "TM": bundle.AntTestTMEnvCfg,
}

for _key, _base in BUNDLE_TEST_ENVS.items():
    globals()[f"AntRMABundle{_key}EnvCfg"] = _with_our_sensors(_base, f"AntRMABundle{_key}EnvCfg")


@configclass
class AntRMABundleDRRoughEnvCfg(bundle.AntDRRoughEnvCfg):
    """[ant_robust_bundle] Training env for our method on the bundle's training terrain: Isaac-Ant-DR-Rough-v0
    unchanged (terrain, domain randomization, terminations, rewards) + our sensors and observation groups.
    Observation noise on the height scan and contacts as in our own training (Isaac-Ant-RMA-v0)."""

    scene: AntRMABundleSceneCfg = AntRMABundleSceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)
    observations: AntRMAObservationsCfg = AntRMAObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        for group in (self.observations.policy, self.observations.teacher):
            group.height_scan.params["plane_height"] = None
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
