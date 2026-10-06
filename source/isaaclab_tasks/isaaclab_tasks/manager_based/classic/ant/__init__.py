# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment (similar to OpenAI Gym Ant-v2).
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Ant-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_env_cfg:AntEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

##
# Generalization to unseen terrains (reward and terminations unchanged).
##

gym.register(
    id="Isaac-Ant-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:AntRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughPPORunnerCfg",
        "rsl_rl_warm_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughWarmPPORunnerCfg",
        "rsl_rl_gated_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntGatedPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Ant-Rough-Rehearsal-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:AntRoughRehearsalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughPPORunnerCfg",
        "rsl_rl_warm_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughWarmPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Ant-Rough-Test-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:AntRoughTestEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughPPORunnerCfg",
        "rsl_rl_warm_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughWarmPPORunnerCfg",
        "rsl_rl_gated_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntGatedPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Ant-Test-Blind-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:AntTestBlindEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
        # for blind policies trained with observation normalization (Isaac-Ant-Rough-Blind-v0)
        "rsl_rl_norm_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughBlindPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Ant-Rough-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:AntRoughFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughPPORunnerCfg",
        "rsl_rl_warm_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughWarmPPORunnerCfg",
        "rsl_rl_gated_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntGatedPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Ant-Rough-Blind-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:AntRoughBlindEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughBlindPPORunnerCfg",
        "rsl_rl_warm_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughBlindWarmPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Ant-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:AntBlindFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
        "rsl_rl_norm_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughBlindPPORunnerCfg",
    },
)

##
# Teacher-student with foot contact sensors (ground-property estimation)
##

gym.register(
    id="Isaac-Ant-RMA-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_rma_env_cfg:AntRMAEnvCfg",
        "rsl_rl_finetune_safe_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMAFinetuneSafePPORunnerCfg",
        "rsl_rl_distillation_long_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMADistillationLongRunnerCfg",
        # student PPO fine-tuning (asymmetric actor-critic), also used to evaluate the fine-tuned student
        "rsl_rl_finetune_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMAFinetunePPORunnerCfg",
        # teacher (PPO, privileged observations)
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMATeacherPPORunnerCfg",
        # student distillation
        "rsl_rl_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMADistillationRunnerCfg",
        # evaluation of the distilled student, alone or inside the terrain gate
        "rsl_rl_student_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMAStudentPPORunnerCfg",
        "rsl_rl_student_gated_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMAStudentGatedPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Ant-RMA-Test-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_rma_env_cfg:AntRMATestEnvCfg",
        "rsl_rl_finetune_safe_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMAFinetuneSafePPORunnerCfg",
        "rsl_rl_distillation_long_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMADistillationLongRunnerCfg",
        # student PPO fine-tuning (asymmetric actor-critic), also used to evaluate the fine-tuned student
        "rsl_rl_finetune_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMAFinetunePPORunnerCfg",
        # teacher (PPO, privileged observations)
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMATeacherPPORunnerCfg",
        # student distillation
        "rsl_rl_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMADistillationRunnerCfg",
        # evaluation of the distilled student, alone or inside the terrain gate
        "rsl_rl_student_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMAStudentPPORunnerCfg",
        "rsl_rl_student_gated_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMAStudentGatedPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Ant-RMA-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_rma_env_cfg:AntRMAFlatEnvCfg",
        "rsl_rl_finetune_safe_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMAFinetuneSafePPORunnerCfg",
        "rsl_rl_distillation_long_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMADistillationLongRunnerCfg",
        # student PPO fine-tuning (asymmetric actor-critic), also used to evaluate the fine-tuned student
        "rsl_rl_finetune_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMAFinetunePPORunnerCfg",
        # teacher (PPO, privileged observations)
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMATeacherPPORunnerCfg",
        # student distillation
        "rsl_rl_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMADistillationRunnerCfg",
        # evaluation of the distilled student, alone or inside the terrain gate
        "rsl_rl_student_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMAStudentPPORunnerCfg",
        "rsl_rl_student_gated_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMAStudentGatedPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Ant-RMA-Wide-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_rma_env_cfg:AntRMAWideEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMATeacherPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMADistillationRunnerCfg",
        "rsl_rl_student_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMAStudentPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Ant-RMA-Wide-Test-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_rma_env_cfg:AntRMAWideTestEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMATeacherPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMADistillationRunnerCfg",
        "rsl_rl_student_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMAStudentPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Ant-RMA-Wide-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_rma_env_cfg:AntRMAWideFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMATeacherPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMADistillationRunnerCfg",
        "rsl_rl_student_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMAStudentPPORunnerCfg",
    },
)






##
# [ant_robust_bundle] Teammate robustness experiment, merged from ant_robust_bundle.zip (2026-10-06).
# Environments: ant_robust_env_cfg.py. Checkpoints: checkpoints/ant_robust_bundle/. Same 60-D observation and PPO
# config as Isaac-Ant-v0. The Isaac-Ant-RMA-Bundle-* tasks further below are our sensor variants of its test envs.
##

_ROBUST_TASKS = {
    # training variants
    "Isaac-Ant-DR-v0": "AntDREnvCfg",
    "Isaac-Ant-DR-Rough-v0": "AntDRRoughEnvCfg",
    "Isaac-Ant-DR-Rough-PushNoise-v0": "AntDRRoughPushNoiseEnvCfg",
    "Isaac-Ant-DR-Rough-Smooth-v0": "AntDRRoughSmoothEnvCfg",
    "Isaac-Ant-DR-Rough-PushNoise-Smooth-v0": "AntDRRoughPushNoiseSmoothEnvCfg",
    # unseen evaluation variants
    "Isaac-Ant-Test-LowFriction-v0": "AntTestLowFrictionEnvCfg",
    "Isaac-Ant-Test-Heavy-v0": "AntTestHeavyEnvCfg",
    "Isaac-Ant-Test-WeakMotor-v0": "AntTestWeakMotorEnvCfg",
    "Isaac-Ant-Test-Rough-v0": "AntTestRoughEnvCfg",
    "Isaac-Ant-Test-Push-v0": "AntTestPushEnvCfg",
    "Isaac-Ant-Test-Heading-v0": "AntTestHeadingEnvCfg",
    "Isaac-Ant-Test-Combined-v0": "AntTestCombinedEnvCfg",
    # teammate-terrain reconstruction (training variants + unseen evaluation)
    "Isaac-Ant-DR-TM-v0": "AntDRTMEnvCfg",
    "Isaac-Ant-DR-TM-PushNoise-v0": "AntDRTMPushNoiseEnvCfg",
    "Isaac-Ant-DR-TM-Smooth-v0": "AntDRTMSmoothEnvCfg",
    "Isaac-Ant-DR-TM-PushNoise-Smooth-v0": "AntDRTMPushNoiseSmoothEnvCfg",
    "Isaac-Ant-Test-TM-v0": "AntTestTMEnvCfg",
}

for _task_id, _cfg_class in _ROBUST_TASKS.items():
    gym.register(
        id=_task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_robust_env_cfg:{_cfg_class}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
        },
    )


# [ant_robust_bundle] Our sensor policy in the bundle's test envs (ant_rma_bundle_eval_cfg.py)
for _key in ["LowFriction", "Heavy", "WeakMotor", "Rough", "Push", "Heading", "Combined", "TM"]:
    gym.register(
        id=f"Isaac-Ant-RMA-Bundle-{_key}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_rma_bundle_eval_cfg:AntRMABundle{_key}EnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMATeacherPPORunnerCfg",
            "rsl_rl_student_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMAStudentPPORunnerCfg",
            "rsl_rl_finetune_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMAFinetunePPORunnerCfg",
        },
    )

# [ant_robust_bundle] Training env for our method on the bundle's training terrain (Isaac-Ant-DR-Rough-v0 + our sensors)
gym.register(
    id="Isaac-Ant-RMA-Bundle-DR-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_rma_bundle_eval_cfg:AntRMABundleDRRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMATeacherPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMADistillationRunnerCfg",
        "rsl_rl_finetune_safe_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMAFinetuneSafePPORunnerCfg",
        "rsl_rl_finetune_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMAFinetunePPORunnerCfg",
    },
)

##
# A new evaluation environment (ant_new_env_cfg.py): one definition, registered for the baseline and for our policy
##

gym.register(
    id="Isaac-Ant-New-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_new_env_cfg:AntNewEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Ant-RMA-New-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_new_env_cfg:AntRMANewEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMATeacherPPORunnerCfg",
        "rsl_rl_finetune_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRMAFinetunePPORunnerCfg",
    },
)

##
# Make the terrain-gated policy class visible to the RSL-RL runner (it resolves ``class_name`` in its own namespace).
##

try:
    import rsl_rl.runners.on_policy_runner as _on_policy_runner

    from .gated_actor_critic import TerrainGatedActorCritic

    _on_policy_runner.TerrainGatedActorCritic = TerrainGatedActorCritic
except ImportError:
    pass
