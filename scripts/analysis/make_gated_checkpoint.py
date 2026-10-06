"""Combine a height-scan terrain expert and the original Isaac-Ant-v0 policy into one TerrainGatedActorCritic checkpoint.

    actor.expert.*  <- expert checkpoint actor.*      (obs: 60 original + 273 height scan)
    actor.flat.*    <- original checkpoint actor.*    (obs: 60 original), frozen in the gated policy
    critic.*, std   <- expert checkpoint

Example:
    python scripts/analysis/make_gated_checkpoint.py \
        --expert logs/rsl_rl/ant_rough/<v4 run>/model_1999.pt \
        --flat logs/rsl_rl/ant/2026-09-28_14-21-51_ant_baseline/model_999.pt \
        --dst logs/rsl_rl/ant_rough/gated_v4/model_0.pt
"""

import argparse
import os

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--expert", required=True)
parser.add_argument("--flat", required=True)
parser.add_argument("--dst", required=True)
parser.add_argument("--std", type=float, default=None, help="Optionally reset the exploration noise.")
args = parser.parse_args()

expert = torch.load(args.expert, map_location="cpu", weights_only=False)
flat = torch.load(args.flat, map_location="cpu", weights_only=False)
sd = {}
for k, v in expert["model_state_dict"].items():
    sd[k.replace("actor.", "actor.expert.", 1) if k.startswith("actor.") else k] = v
for k, v in flat["model_state_dict"].items():
    if k.startswith("actor."):
        sd[k.replace("actor.", "actor.flat.", 1)] = v
if args.std is not None:
    sd["std"] = torch.full_like(sd["std"], args.std)

opt = expert["optimizer_state_dict"]
opt["state"] = {}
num_params = sum(1 for k in sd if not k.startswith(("actor_obs_normalizer", "critic_obs_normalizer")))
opt["param_groups"][0]["params"] = list(range(num_params))
ckpt = {
    "model_state_dict": sd,
    "optimizer_state_dict": opt,
    "iter": 0,
    "infos": {"expert": args.expert, "flat": args.flat},
}
os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
torch.save(ckpt, args.dst)
for k, v in sd.items():
    print(f"{k:28s} {tuple(v.shape)}")
print("saved", args.dst)
