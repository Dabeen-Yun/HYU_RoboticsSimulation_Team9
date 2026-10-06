"""Checkpoint conversions for the teacher-student (Isaac-Ant-RMA-*) experiment.

init:     teacher PPO checkpoint -> distillation checkpoint whose student starts as the teacher without the
          privileged inputs (the privileged observations are the last ``--num_privileged`` inputs of the teacher).
export:   distillation checkpoint -> (a) student as a plain ActorCritic, (b) student inside the terrain gate with the
          original Isaac-Ant-v0 policy on flat ground. The critic is not used at evaluation and is zero-filled.

Examples:
    python scripts/analysis/make_student_checkpoints.py init --teacher logs/rsl_rl/ant_rma/<teacher>/model_1999.pt \
        --dst logs/rsl_rl/ant_rma/student_init/model_0.pt
    python scripts/analysis/make_student_checkpoints.py export --student logs/rsl_rl/ant_rma/<student>/model_1499.pt \
        --flat logs/rsl_rl/ant/2026-09-28_14-21-51_ant_baseline/model_999.pt --dst_dir logs/rsl_rl/ant_rma
"""

import argparse
import copy
import os

import torch

parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest="cmd", required=True)
p_init = sub.add_parser("init")
p_init.add_argument("--teacher", required=True)
p_init.add_argument("--dst", required=True)
p_init.add_argument("--num_privileged", type=int, default=3)
p_init.add_argument("--std", type=float, default=0.05)
p_ft = sub.add_parser("finetune_init")
p_ft.add_argument("--student", required=True, help="distillation checkpoint (student.*)")
p_ft.add_argument("--teacher", required=True, help="teacher PPO checkpoint (critic.* is reused)")
p_ft.add_argument("--dst", required=True)
p_ft.add_argument("--std", type=float, default=0.05)
p_exp = sub.add_parser("export")
p_exp.add_argument("--student", required=True)
p_exp.add_argument("--flat", required=True)
p_exp.add_argument("--dst_dir", required=True)
args = parser.parse_args()


def _save(sd, template_opt, path, info):
    opt = copy.deepcopy(template_opt)
    opt["state"] = {}
    opt["param_groups"] = opt["param_groups"][:1]
    opt["param_groups"][0]["params"] = list(range(sum(1 for k in sd if "normalizer" not in k)))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"model_state_dict": sd, "optimizer_state_dict": opt, "iter": 0, "infos": info}, path)
    print("saved", path)


def _actor_layers(sd, prefix):
    return {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}


if args.cmd == "init":
    ckpt = torch.load(args.teacher, map_location="cpu", weights_only=False)
    actor = _actor_layers(ckpt["model_state_dict"], "actor.")
    sd = {}
    for k, v in actor.items():
        sd["teacher." + k] = v
        sd["student." + k] = v[:, : -args.num_privileged].clone() if k == "0.weight" else v.clone()
    sd["std"] = torch.full_like(ckpt["model_state_dict"]["std"], args.std)
    print("student.0.weight", tuple(sd["student.0.weight"].shape), "teacher.0.weight", tuple(sd["teacher.0.weight"].shape))
    _save(sd, ckpt["optimizer_state_dict"], args.dst, {"student_init_from_teacher": args.teacher})

elif args.cmd == "finetune_init":
    student_ckpt = torch.load(args.student, map_location="cpu", weights_only=False)
    teacher_ckpt = torch.load(args.teacher, map_location="cpu", weights_only=False)
    sd = {"actor." + k: v for k, v in _actor_layers(student_ckpt["model_state_dict"], "student.").items()}
    sd.update({k: v for k, v in teacher_ckpt["model_state_dict"].items() if k.startswith("critic.")})
    sd["std"] = torch.full_like(student_ckpt["model_state_dict"]["std"], args.std)
    print("actor.0.weight", tuple(sd["actor.0.weight"].shape), "critic.0.weight", tuple(sd["critic.0.weight"].shape))
    _save(sd, teacher_ckpt["optimizer_state_dict"], args.dst, {"student": args.student, "critic_from": args.teacher})

else:
    ckpt = torch.load(args.student, map_location="cpu", weights_only=False)
    student = _actor_layers(ckpt["model_state_dict"], "student.")
    flat = _actor_layers(torch.load(args.flat, map_location="cpu", weights_only=False)["model_state_dict"], "actor.")
    std = ckpt["model_state_dict"]["std"]
    # zero critic with the actor's layer shapes (input = student observations, output = 1)
    critic = {k: torch.zeros_like(v) for k, v in student.items()}
    last_w = max((k for k in critic if k.endswith("weight")), key=lambda k: int(k.split(".")[0]))
    critic[last_w] = torch.zeros(1, critic[last_w].shape[1])
    critic[last_w.replace("weight", "bias")] = torch.zeros(1)
    base = {"critic." + k: v for k, v in critic.items()}
    base["std"] = std

    plain = dict(base)
    plain.update({"actor." + k: v for k, v in student.items()})
    _save(plain, ckpt["optimizer_state_dict"], os.path.join(args.dst_dir, "student_plain", "model_0.pt"), {"student": args.student})

    gated = dict(base)
    gated.update({"actor.expert." + k: v for k, v in student.items()})
    gated.update({"actor.flat." + k: v for k, v in flat.items()})
    _save(gated, ckpt["optimizer_state_dict"], os.path.join(args.dst_dir, "student_gated", "model_0.pt"),
          {"student": args.student, "flat": args.flat})
