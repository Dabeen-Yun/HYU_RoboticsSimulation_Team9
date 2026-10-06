# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Detailed evaluation of an RSL-RL Ant checkpoint.

Runs the same loop as ``scripts/reinforcement_learning/rsl_rl/play_one_episode.py`` (first episode of every env,
same seed handling), so the reward total mean/std is identical. In addition it records, per env:
    * the reward of each reward term
    * the terrain type (sub-terrain column) and the start/end position
    * the root xy trajectory
and saves everything to an ``.npz`` file for plotting.

Example:
    ./isaaclab.sh -p scripts/analysis/evaluate_ant.py --task Isaac-Ant-Rough-Test-v0 --seed 24 --num_envs 100 \
        --headless --checkpoint logs/rsl_rl/ant_rough/<run>/model_2999.pt --out results/eval/ours_test.npz
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "reinforcement_learning", "rsl_rl"))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Detailed one-episode evaluation of an RSL-RL Ant policy.")
parser.add_argument("--num_envs", type=int, default=100)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=24)
parser.add_argument("--out", type=str, required=True, help="Output .npz path.")
parser.add_argument("--label", type=str, default="", help="Label stored in the output file.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


def _terrain_type_names(env_cfg: ManagerBasedRLEnvCfg) -> list[str]:
    """Name of the sub-terrain in each column (same allocation rule as TerrainGenerator)."""
    gen = env_cfg.scene.terrain.terrain_generator
    if gen is None:
        return ["plane"]
    names = list(gen.sub_terrains.keys())
    proportions = np.array([c.proportion for c in gen.sub_terrains.values()])
    proportions /= proportions.sum()
    return [names[int(np.min(np.where(i / gen.num_cols + 0.001 < np.cumsum(proportions))[0]))] for i in range(gen.num_cols)]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    uenv = env.unwrapped

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(retrieve_file_path(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=uenv.device)

    robot = uenv.scene["robot"]
    rm = uenv.reward_manager
    n = env.num_envs
    dt = uenv.step_dt

    terrain = uenv.scene.terrain
    if getattr(terrain, "terrain_types", None) is not None:
        terrain_col = terrain.terrain_types.cpu().numpy()
    else:
        terrain_col = np.zeros(n, dtype=np.int64)

    obs = env.get_observations()
    episode_rewards = torch.zeros(n, dtype=torch.float64, device=uenv.device)
    term_rewards = torch.zeros(n, len(rm.active_terms), dtype=torch.float64, device=uenv.device)
    episode_steps = torch.zeros(n, dtype=torch.long, device=uenv.device)
    finished = torch.zeros(n, dtype=torch.bool, device=uenv.device)
    timed_out = torch.zeros(n, dtype=torch.bool, device=uenv.device)
    start_pos = robot.data.root_pos_w.clone()
    end_pos = start_pos.clone()
    traj = [start_pos[:, :2].cpu().numpy()]

    timestep = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            pos_before = robot.data.root_pos_w.clone()
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)
            active = ~finished
            episode_rewards[active] += rewards[active]
            term_rewards[active] += rm._step_reward[active].double() * dt
            episode_steps[active] += 1
            done = dones.bool()
            # position after the step, or the last pre-reset position for envs that just ended
            pos_now = torch.where(done.unsqueeze(-1), pos_before, robot.data.root_pos_w)
            end_pos[active] = pos_now[active]
            newly = active & done
            timed_out[newly] = uenv.termination_manager.time_outs[newly]
            finished |= done
            traj.append(end_pos[:, :2].cpu().numpy())
        timestep += 1
        if finished.all().item() or timestep >= uenv.max_episode_length:
            break

    totals = episode_rewards.cpu().numpy()
    print(f"[RESULT] {args_cli.task}: reward mean={totals.mean():.6f}, std={totals.std():.6f}")
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    np.savez(
        args_cli.out,
        label=args_cli.label,
        task=args_cli.task,
        checkpoint=args_cli.checkpoint,
        reward_total=totals,
        term_names=np.array(rm.active_terms),
        term_rewards=term_rewards.cpu().numpy(),
        steps=episode_steps.cpu().numpy(),
        timed_out=timed_out.cpu().numpy(),
        terrain_col=terrain_col,
        terrain_type_names=np.array(_terrain_type_names(env_cfg)),
        start_pos=start_pos.cpu().numpy(),
        end_pos=end_pos.cpu().numpy(),
        traj=np.stack(traj, axis=1),
        dt=dt,
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
