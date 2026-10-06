# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export height maps of the Ant terrains (train / test / flat) for plotting.

The terrain mesh is generated exactly as in the environments and ray-cast from above on a regular grid
(the same operation the height scanner performs).

Example:
    ./isaaclab.sh -p scripts/analysis/export_terrains.py --headless --out results/terrains.npz
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export Ant terrain height maps.")
parser.add_argument("--out", type=str, default="results/terrains.npz")
parser.add_argument("--resolution", type=float, default=0.1)
parser.add_argument("--train_seed", type=int, default=0, help="Seed for the (otherwise unseeded) training terrain.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import os

import numpy as np
import torch

from isaaclab.terrains import TerrainGenerator
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

from isaaclab_tasks.manager_based.classic.ant.ant_rough_env_cfg import ANT_TEST_TERRAINS_CFG, ANT_TRAIN_TERRAINS_CFG


def export(cfg, plane_height=None, device="cuda:0"):
    gen = TerrainGenerator(cfg, device=device)
    mesh = gen.terrain_mesh
    wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=device)
    # only the sub-terrain grid (+1 m margin), not the whole border
    half_x = cfg.num_rows * cfg.size[0] / 2 + 1.0
    half_y = cfg.num_cols * cfg.size[1] / 2 + 1.0
    xs = np.arange(-half_x, half_x, args_cli.resolution)
    ys = np.arange(-half_y, half_y, args_cli.resolution)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    starts = torch.tensor(np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, 50.0)], -1), dtype=torch.float32, device=device)
    dirs = torch.zeros_like(starts)
    dirs[:, 2] = -1.0
    hits = raycast_mesh(starts, dirs, wp_mesh)[0]
    heights = hits[:, 2].reshape(gx.shape).cpu().numpy()
    if plane_height is not None:
        # the training scene also has the original ground plane at z = 0
        heights = np.where(np.isfinite(heights), np.maximum(heights, plane_height), plane_height)
    names = list(cfg.sub_terrains.keys())
    proportions = np.array([c.proportion for c in cfg.sub_terrains.values()])
    proportions /= proportions.sum()
    col_types = [names[int(np.min(np.where(i / cfg.num_cols + 0.001 < np.cumsum(proportions))[0]))] for i in range(cfg.num_cols)]
    return dict(heights=heights, xs=xs, ys=ys, origins=gen.terrain_origins, col_types=np.array(col_types))


def main():
    out = {}
    train_cfg = ANT_TRAIN_TERRAINS_CFG.copy()
    train_cfg.seed = args_cli.train_seed
    for key, cfg, plane in [("train", train_cfg, 0.0), ("test", ANT_TEST_TERRAINS_CFG, None)]:
        d = export(cfg, plane)
        for k, v in d.items():
            out[f"{key}_{k}"] = v
        print(f"[INFO] {key}: heights {d['heights'].shape}, min={np.nanmin(d['heights']):.3f}, max={np.nanmax(d['heights']):.3f}")
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    np.savez_compressed(args_cli.out, **out)
    print(f"[INFO] saved {args_cli.out}")


if __name__ == "__main__":
    main()
    simulation_app.close()
