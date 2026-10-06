# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain-gated actor-critic for RSL-RL.

The actor combines two experts and switches between them using the height scan:
    * ``flat``:   the original Isaac-Ant-v0 policy (input: the 60 original observations), frozen.
    * ``expert``: the terrain policy (input: original observations + height scan).
The gate is the height range (max - min) of the height scan. On flat ground it is exactly 0 and the original policy
acts, so performance on the original task is preserved by construction; as soon as the scan sees relief above
``gate_threshold`` the terrain expert acts.

The gating lives inside ``policy.actor`` so the standard RSL-RL runner, ``play.py``/``play_one_episode.py`` and the
JIT/ONNX exporters work unchanged.
"""

import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP


class GatedActor(nn.Module):
    def __init__(self, expert: nn.Module, flat: nn.Module, flat_obs_dim: int, gate_threshold: float, scan_dim: int = -1):
        super().__init__()
        self.expert = expert
        self.flat = flat
        self.flat_obs_dim = flat_obs_dim
        self.gate_threshold = gate_threshold
        # the height scan follows the original observations; -1: it runs to the end of the observation vector
        self.scan_end = -1 if scan_dim < 0 else flat_obs_dim + scan_dim

    def gate(self, obs: torch.Tensor) -> torch.Tensor:
        """1 where the terrain expert acts, 0 where the original (flat) policy acts."""
        scan = obs[:, self.flat_obs_dim :] if self.scan_end < 0 else obs[:, self.flat_obs_dim : self.scan_end]
        relief = scan.max(dim=-1).values - scan.min(dim=-1).values
        return (relief > self.gate_threshold).to(obs.dtype).unsqueeze(-1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        g = self.gate(obs)
        return g * self.expert(obs) + (1.0 - g) * self.flat(obs[:, : self.flat_obs_dim])

    def __getitem__(self, idx):
        # the ONNX exporter reads ``actor[0].in_features``
        return self.expert[idx]


class TerrainGatedActorCritic(ActorCritic):
    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        flat_obs_dim: int = 60,
        flat_hidden_dims=(400, 200, 100),
        gate_threshold: float = 0.005,
        scan_dim: int = -1,
        train_flat: bool = False,
        **kwargs,
    ):
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        activation = kwargs.get("activation", "elu")
        flat = MLP(flat_obs_dim, num_actions, list(flat_hidden_dims), activation)
        if not train_flat:
            for p in flat.parameters():
                p.requires_grad_(False)
        self.actor = GatedActor(self.actor, flat, flat_obs_dim, gate_threshold, scan_dim)
        print(f"[TerrainGatedActorCritic] flat policy on obs[:{flat_obs_dim}], gate threshold {gate_threshold} m")
