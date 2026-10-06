"""Build a warm-start checkpoint for the height-scan Ant policy from the original (blind) Isaac-Ant-v0 policy.

The height-scan policy observes the 60 original observation terms (same order, same scales) followed by the
273 height-scan values. The first layer of the actor and critic is widened with zero weights for the new inputs,
so the warm-started policy initially behaves exactly like the original one. The exploration noise is reset to
``--std`` because the original policy's noise has collapsed (~0.01-0.05), which would prevent learning to use
the new inputs. The optimizer state is dropped (shapes changed) and the iteration counter is reset.

Example:
    python scripts/analysis/make_warmstart_checkpoint.py \
        --src logs/rsl_rl/ant/2026-09-28_14-21-51_ant_baseline/model_999.pt \
        --dst logs/rsl_rl/ant_rough/warmstart_from_baseline/model_0.pt --new_inputs 273 --std 0.2
"""

import argparse
import os

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True)
parser.add_argument("--dst", required=True)
parser.add_argument("--new_inputs", type=int, default=273)
parser.add_argument("--std", type=float, default=0.2)
args = parser.parse_args()

ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
sd = ckpt["model_state_dict"]
for net in ("actor", "critic"):
    w = sd[f"{net}.0.weight"]
    sd[f"{net}.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], args.new_inputs, dtype=w.dtype)], dim=1)
    print(f"{net}.0.weight: {tuple(w.shape)} -> {tuple(sd[f'{net}.0.weight'].shape)}")
print(f"std: {sd['std'].numpy().round(3)} -> {args.std}")
sd["std"] = torch.full_like(sd["std"], args.std)

ckpt["optimizer_state_dict"]["state"] = {}
ckpt["iter"] = 0
ckpt["infos"] = {"warm_start_from": args.src}
os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
torch.save(ckpt, args.dst)
print("saved", args.dst)
