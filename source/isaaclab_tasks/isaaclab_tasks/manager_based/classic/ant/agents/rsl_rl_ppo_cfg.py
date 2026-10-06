# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class AntPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 1000
    save_interval = 50
    experiment_name = "ant"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[400, 200, 100],
        critic_hidden_dims=[400, 200, 100],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )



@configclass
class AntRoughPPORunnerCfg(AntPPORunnerCfg):
    max_iterations = 1500
    experiment_name = "ant_rough"
    # observations mix proprioception and a 273-dim height map with different scales -> normalize them
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[400, 200, 100],
        critic_hidden_dims=[400, 200, 100],
        activation="elu",
    )


@configclass
class AntRoughBlindPPORunnerCfg(AntRoughPPORunnerCfg):
    experiment_name = "ant_rough_blind"


@configclass
class AntRoughWarmPPORunnerCfg(AntPPORunnerCfg):
    """Height-scan policy warm-started from the original Isaac-Ant-v0 policy (see make_warmstart_checkpoint.py).
    No observation normalization, like the original policy, so that its weights stay valid."""

    max_iterations = 2000
    experiment_name = "ant_rough"


@configclass
class RslRlGatedActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "TerrainGatedActorCritic"
    flat_obs_dim: int = 60
    flat_hidden_dims: list[int] = [400, 200, 100]
    gate_threshold: float = 0.005
    """Height range (max - min) of the height scan above which the terrain expert acts [m]."""
    scan_dim: int = -1
    """Number of height-scan values after the original observations (-1: until the end of the observation)."""
    train_flat: bool = False


@configclass
class AntGatedPPORunnerCfg(AntRoughWarmPPORunnerCfg):
    """Terrain-gated policy: frozen original Isaac-Ant-v0 policy on flat ground + height-scan terrain expert."""

    experiment_name = "ant_rough"
    policy = RslRlGatedActorCriticCfg(
        init_noise_std=0.05,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[400, 200, 100],
        critic_hidden_dims=[400, 200, 100],
        activation="elu",
    )


@configclass
class AntRoughBlindWarmPPORunnerCfg(AntRoughWarmPPORunnerCfg):
    """Ablation of the warm-started terrain expert without height scan."""

    experiment_name = "ant_rough_blind"


##
# Teacher-student (Isaac-Ant-RMA-*)
##


@configclass
class AntRMATeacherPPORunnerCfg(AntRoughWarmPPORunnerCfg):
    """Teacher: PPO on the privileged observations, warm-started from the v4 terrain expert."""

    max_iterations = 2000
    experiment_name = "ant_rma"
    obs_groups = {"policy": ["teacher"], "critic": ["teacher"]}


@configclass
class AntRMADistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Student: imitates the teacher (DAgger) from deployable observations (contacts + history, no privileged)."""

    num_steps_per_env = 64
    max_iterations = 1500
    save_interval = 100
    experiment_name = "ant_rma"
    obs_groups = {"policy": ["policy"], "teacher": ["teacher"]}
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.05,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[400, 200, 100],
        teacher_hidden_dims=[400, 200, 100],
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
        max_grad_norm=1.0,
    )


@configclass
class AntRMAStudentPPORunnerCfg(AntRoughWarmPPORunnerCfg):
    """Evaluation of the distilled student as a plain actor-critic (see make_student_checkpoints.py)."""

    experiment_name = "ant_rma"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}


@configclass
class AntRMAStudentGatedPPORunnerCfg(AntGatedPPORunnerCfg):
    """Evaluation of the distilled student inside the terrain gate (original policy on flat ground)."""

    experiment_name = "ant_rma"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    policy = RslRlGatedActorCriticCfg(
        init_noise_std=0.05,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[400, 200, 100],
        critic_hidden_dims=[400, 200, 100],
        activation="elu",
        scan_dim=273,
    )


@configclass
class AntRMAFinetunePPORunnerCfg(AntRoughWarmPPORunnerCfg):
    """Student fine-tuning with PPO (asymmetric actor-critic): the actor sees only the deployable student observations,
    the critic (initialized from the teacher's critic) additionally sees the privileged ground properties. Closes the
    imitation gap by optimizing the true reward directly. The critic is used only during training."""

    max_iterations = 1500
    experiment_name = "ant_rma"
    obs_groups = {"policy": ["policy"], "critic": ["teacher"]}


@configclass
class AntRMAFinetuneSafePPORunnerCfg(AntRMAFinetunePPORunnerCfg):
    """Conservative fine-tuning: small fixed learning rate so that the first updates (with a critic that was fitted to
    the teacher, not to the student) cannot destroy the distilled policy."""

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class AntRMADistillationLongRunnerCfg(AntRMADistillationRunnerCfg):
    """Continued distillation with a lower learning rate."""

    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=3.0e-4,
        gradient_length=15,
        max_grad_norm=1.0,
    )
