"""Plot the Ant generalization experiment (plain Python, no simulator needed).

Inputs (produced by export_terrains.py / evaluate_ant.py / training logs):
    results/terrains.npz
    results/eval/<policy>_<env>.npz      policy in {baseline, blind_rough, ours}, env in {flat, test}
    logs/rsl_rl/<experiment>/<run>/events.out.tfevents.*

Outputs: results/figures/*.png

Example:
    python scripts/analysis/plot_results.py
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

parser = argparse.ArgumentParser()
parser.add_argument("--results", default="results")
parser.add_argument("--baseline_run", default="logs/rsl_rl/ant/2026-09-28_14-21-51_ant_baseline")
parser.add_argument("--ours_run", default=None, help="Defaults to the latest logs/rsl_rl/ant_rough/*")
parser.add_argument("--blind_run", default=None, help="Defaults to the latest logs/rsl_rl/ant_rough_blind/*")
args = parser.parse_args()

# ---------------------------------------------------------------- style
for path in glob.glob("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
    font_manager.fontManager.addfont(path)
plt.rcParams.update(
    {
        "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#8a8984",
        "axes.labelcolor": "#52514e",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "text.color": "#0b0b0b",
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "legend.frameon": False,
    }
)
TEXT2 = "#52514e"
# fixed categorical slots (validated reference palette, light mode): colour follows the policy, never the rank
POLICIES = {
    "student": dict(label="Ours: 교사-학생 (높이맵 + 발 접촉 + 이력)", short="Ours (학생)", color="#2a78d6"),
    "baseline": dict(label="Baseline: 원본 Isaac-Ant-v0", short="Baseline", color="#eb6834"),
    "blind_warm": dict(label="높이맵 없음 (원본에서 이어 학습)", short="높이맵 없음", color="#1baf7a"),
    "scan_scratch": dict(label="높이맵, 처음부터 학습 (v3)", short="높이맵·처음부터", color="#eda100"),
    "scan_warm": dict(label="높이맵, 원본에서 이어 학습 (v4)", short="높이맵(v4)", color="#e87ba4"),
    "scan_rehearsal": dict(label="높이맵, 이어 학습 + 리허설 (v5)", short="높이맵·리허설", color="#008300"),
    "gated_v4": dict(label="높이맵 + 전문가 전환 (이전 최종)", short="전문가 전환", color="#4a3aa7"),
    "teacher": dict(label="교사 (정답 물성 관측, 배포 불가)", short="교사", color="#e34948"),
}
# variant of Ours: same colour family, dashed line (friction sweep only)
VARIANTS = {
    "student_wide": dict(label="Ours 변형: 넓은 범위 학습 (마찰 0.1~2.0 등)", short="Ours 넓은 범위", color="#2a78d6", ls="--"),
    "student_imu": dict(label="Ours 변형: + 몸통 IMU (중력 방향)", short="Ours + IMU", color="#2a78d6", ls=":"),
}
# entities used in the busier plots
MAIN = ["baseline", "blind_warm", "student"]
TERRAIN_POLICIES = ["baseline", "blind_warm", "scan_warm", "student"]
ENVS = {"flat": "기존 환경 (평지)", "test": "처음 보는 환경"}
TERRAIN_KO = {
    "plane": "평면",
    "flat": "평지 (평면 바닥)",
    "random_rough": "요철",
    "pyramid_slope": "경사",
    "pyramid_stairs": "계단",
    "boxes": "박스",
    "wave": "파도 (미학습)",
    "discrete_obstacles": "장애물 (미학습)",
    "random_rough_hard": "거친 요철 (강화)",
}

R = args.results
FIG = os.path.join(R, "figures")
os.makedirs(FIG, exist_ok=True)


def load_eval(policy, env):
    path = os.path.join(R, "eval", f"{policy}_{env}.npz")
    return np.load(path, allow_pickle=True) if os.path.exists(path) else None


def latest(pattern):
    runs = sorted(glob.glob(pattern))
    return runs[-1] if runs else None


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("[saved]", os.path.join(FIG, name))


# ---------------------------------------------------------------- 1. terrains
def plot_terrains():
    path = os.path.join(R, "terrains.npz")
    if not os.path.exists(path):
        return
    t = np.load(path, allow_pickle=True)
    # the original Isaac-Ant-v0 ground is an infinite plane at z = 0
    t = dict(t)
    t["flat_heights"] = np.zeros_like(t["test_heights"])
    t["flat_xs"], t["flat_ys"] = t["test_xs"], t["test_ys"]
    t["flat_col_types"] = np.array(["plane"] * len(t["test_col_types"]))
    keys = [("flat", "기존 환경: Isaac-Ant-v0 (무한 평면 바닥, 같은 축척으로 표시)"),
            ("train", "학습 환경: Isaac-Ant-Rough-v0 (평지 열 = 원래 평면 바닥)"),
            ("test", "처음 보는 환경: Isaac-Ant-Rough-Test-v0 (학습에 쓰지 않음, 지형 seed 24)")]
    fig, axes = plt.subplots(len(keys), 1, figsize=(12, 13), constrained_layout=True)
    for ax, (k, title) in zip(axes, keys):
        h = t[f"{k}_heights"]
        xs, ys = t[f"{k}_xs"], t[f"{k}_ys"]
        hmax = float(np.nanmax(np.where(np.isfinite(h), h, np.nan)))
        vmax = max(0.05, hmax)
        im = ax.imshow(h.T, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap="Greys", vmin=min(-0.02, float(np.nanmin(h))),
                       vmax=vmax, aspect="equal", interpolation="nearest")
        ax.set_title(title, loc="left", fontsize=12, pad=8)
        ax.set_xlabel("x [m]  →  목표 방향 (1000, 0, 0)")
        ax.set_ylabel("y [m]")
        ax.grid(False)
        col_types = t[f"{k}_col_types"]
        ncol = len(col_types)
        y0, y1 = ys[0] + 1.0, ys[-1] - 1.0
        for i, name in enumerate(col_types):
            yc = y0 + (i + 0.5) * (y1 - y0) / ncol
            ax.text(xs[-1] + 1.0, yc, TERRAIN_KO.get(str(name), str(name)), va="center", fontsize=8, color=TEXT2)
        ax.axvline(xs[0] + 5.0, color="#2a78d6", lw=1.2, ls=":")
        ax.text(xs[0] + 6.0, ys[0] + 2.0, "출발선 (모든 개체가 row 0에서 출발)", fontsize=8, color="#2a78d6",
                bbox=dict(facecolor="#fcfcfb", edgecolor="none", alpha=0.85, pad=1.5))
        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.12)
        cb.set_label("지형 높이 [m]")
    save(fig, "fig1_terrains.png")

    # close-up of each sub-terrain type (difficulty ~ middle row)
    for k, title in [("train", "학습 지형 종류"), ("test", "처음 보는 지형 종류")]:
        h = t[f"{k}_heights"]
        xs, ys = t[f"{k}_xs"], t[f"{k}_ys"]
        col_types = list(t[f"{k}_col_types"])
        uniq = list(dict.fromkeys(col_types))
        fig, axes = plt.subplots(1, len(uniq), figsize=(3.2 * len(uniq), 3.4), constrained_layout=True)
        res = xs[1] - xs[0]
        for ax, name in zip(np.atleast_1d(axes), uniq):
            col = col_types.index(name)
            row = 12  # a moderately hard row
            x0 = int((1.0 + row * 8.0) / res)
            y0 = int((1.0 + col * 8.0) / res)
            patch = h[x0 : x0 + int(8 / res), y0 : y0 + int(8 / res)]
            ax.imshow(patch.T, origin="lower", cmap="Greys", vmin=-0.1, vmax=max(0.15, float(np.nanmax(patch))),
                      extent=[0, 8, 0, 8])
            ax.set_title(TERRAIN_KO.get(str(name), str(name)), fontsize=11)
            ax.grid(False)
            ax.set_xticks([0, 4, 8])
            ax.set_yticks([0, 4, 8])
            ax.text(0.2, 0.3, f"높이 {np.nanmin(patch):.2f}~{np.nanmax(patch):.2f} m", fontsize=8, color="#0b0b0b",
                    bbox=dict(facecolor="#fcfcfb", edgecolor="none", alpha=0.8, pad=2))
        fig.suptitle(f"{title} (8 m × 8 m, 난이도 row 12/20)", x=0.01, ha="left", fontsize=12)
        save(fig, f"fig1_{k}_tiles.png")


# ---------------------------------------------------------------- 2. training curves
def read_tb(run, tag):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    files = glob.glob(os.path.join(run, "events*"))
    if not files:
        return None, None
    ea = EventAccumulator(files[0], size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return None, None
    v = ea.Scalars(tag)
    return np.array([e.step for e in v]), np.array([e.value for e in v])


def smooth(y, k=20):
    if len(y) < k:
        return y
    c = np.cumsum(np.insert(y, 0, 0.0))
    s = (c[k:] - c[:-k]) / k
    return np.concatenate([y[: k - 1] * 0 + s[0], s])


def plot_training():
    runs = {
        "baseline": args.baseline_run,
        "blind_warm": latest("logs/rsl_rl/ant_rough_blind/*_blind_warm"),
        "scan_scratch": latest("logs/rsl_rl/ant_rough/*_v3"),
        "scan_warm": latest("logs/rsl_rl/ant_rough/*_v4_warm"),
        "scan_rehearsal": latest("logs/rsl_rl/ant_rough/*_v5_warm_rehearsal"),
        "teacher": latest("logs/rsl_rl/ant_rma/*_teacher"),
        "student": latest("logs/rsl_rl/ant_rma/*_student"),
    }
    panels = [
        ("Train/mean_reward", "에피소드 누적 보상 (학습 환경)"),
        ("Episode_Reward/progress", "전진 보상 / 초  (≈ 전진 속도 m/s)"),
        ("Train/mean_episode_length", "에피소드 길이 [step] (최대 960)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for ax, (tag, title) in zip(axes, panels):
        for key, run in runs.items():
            if run is None:
                continue
            x, y = read_tb(run, tag)
            if x is None:
                continue
            p = POLICIES[key]
            ax.plot(x, smooth(y), color=p["color"], lw=2, label=p["label"])
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("학습 iteration")
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("학습 곡선 — 각 정책의 자기 학습 환경 기준 (Baseline = 평지, 나머지 = 지형+랜덤화). 이어 학습은 각 시작점 이후 (교사 ← v4, 학생 ← 교사 모방)",
                 x=0.01, ha="left", fontsize=12)
    save(fig, "fig2_training_curves.png")


# ---------------------------------------------------------------- 3. evaluation
def plot_eval_bars():
    fig, ax = plt.subplots(figsize=(13, 5.6), constrained_layout=True)
    keys = [k for k in POLICIES if any(load_eval(k, e) is not None for e in ENVS)]
    width = 0.8 / max(len(keys), 1)
    for i, key in enumerate(keys):
        means, stds = [], []
        for env in ENVS:
            d = load_eval(key, env)
            means.append(np.nan if d is None else d["reward_total"].mean())
            stds.append(np.nan if d is None else d["reward_total"].std())
        xs = np.arange(len(ENVS)) + (i - (len(keys) - 1) / 2) * width
        p = POLICIES[key]
        ax.bar(xs, means, width * 0.92, yerr=stds, color=p["color"], label=p["label"], capsize=2,
               error_kw=dict(ecolor="#8a8984", lw=1))
        for x, m, s in zip(xs, means, stds):
            if np.isfinite(m):
                ax.text(x, m + s + 2, f"{m:.0f}", ha="center", fontsize=8, color="#0b0b0b")
    ax.set_xticks(np.arange(len(ENVS)), list(ENVS.values()))
    ax.set_ylabel("에피소드 누적 보상 (원본 보상 함수)")
    ax.set_title("평가: seed 24, 환경 100개, 첫 에피소드 (평균 ± 표준편차)", loc="left", fontsize=12)
    ref = load_eval("baseline", "flat")["reward_total"].mean()
    ax.axhline(ref, color="#8a8984", lw=1, ls="--")
    ax.text(len(ENVS) - 0.55, ref + 2, f"Baseline 기존 환경 {ref:.1f}", fontsize=8, color=TEXT2, ha="right")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=8)
    ax.grid(axis="x", visible=False)
    save(fig, "fig3_eval_reward.png")


def plot_eval_by_terrain():
    keys = [k for k in TERRAIN_POLICIES if load_eval(k, "test") is not None]
    if not keys:
        return
    names = list(dict.fromkeys(load_eval(keys[0], "test")["terrain_type_names"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    width = 0.8 / len(keys)
    for i, key in enumerate(keys):
        d = load_eval(key, "test")
        types = d["terrain_type_names"][d["terrain_col"]]
        dist = d["end_pos"][:, 0] - d["start_pos"][:, 0]
        r_m, d_m = [], []
        for n in names:
            m = types == n
            r_m.append(d["reward_total"][m].mean())
            d_m.append(dist[m].mean())
        xs = np.arange(len(names)) + (i - (len(keys) - 1) / 2) * width
        p = POLICIES[key]
        axes[0].bar(xs, r_m, width * 0.92, color=p["color"], label=p["label"])
        axes[1].bar(xs, d_m, width * 0.92, color=p["color"], label=p["label"])
    for ax, t in zip(axes, ["누적 보상", "+x 방향 전진 거리 [m]"]):
        ax.set_xticks(np.arange(len(names)), [TERRAIN_KO.get(str(n), str(n)) for n in names])
        ax.set_title(f"처음 보는 환경 — 지형별 {t}", loc="left", fontsize=12)
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.1), ncol=4, fontsize=9)
    save(fig, "fig4_eval_by_terrain.png")


def plot_trajectories():
    tpath = os.path.join(R, "terrains.npz")
    if not os.path.exists(tpath):
        return
    t = np.load(tpath, allow_pickle=True)
    keys = [k for k in MAIN if load_eval(k, "test") is not None]
    fig, axes = plt.subplots(len(keys), 1, figsize=(12, 4.3 * len(keys)), constrained_layout=True, squeeze=False)
    for ax, key in zip(axes[:, 0], keys):
        h, xs, ys = t["test_heights"], t["test_xs"], t["test_ys"]
        ax.imshow(h.T, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap="Greys", vmin=-0.1, vmax=0.4,
                  aspect="equal", interpolation="nearest")
        d = load_eval(key, "test")
        p = POLICIES[key]
        for j in range(d["traj"].shape[0]):
            tr = d["traj"][j]
            ax.plot(tr[:, 0], tr[:, 1], color=p["color"], lw=0.8, alpha=0.7)
            ax.plot(tr[-1, 0], tr[-1, 1], "o", ms=3, color=p["color"], mec="#fcfcfb", mew=0.5)
        dist = d["end_pos"][:, 0] - d["start_pos"][:, 0]
        ax.set_title(f"{p['label']} — 처음 보는 환경에서의 궤적 100개 (평균 전진 {dist.mean():.1f} m)", loc="left",
                     fontsize=11)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(ys[0] - 14, ys[-1] + 14)
        ax.grid(False)
        for i, name in enumerate(t["test_col_types"]):
            yc = ys[0] + 1 + (i + 0.5) * (ys[-1] - ys[0] - 2) / len(t["test_col_types"])
            ax.text(xs[-1] + 1.0, yc, TERRAIN_KO.get(str(name), str(name)), va="center", fontsize=8, color=TEXT2)
    save(fig, "fig5_trajectories_test.png")


def plot_terms():
    rows = []
    for key in MAIN:
        for env in ENVS:
            d = load_eval(key, env)
            if d is not None:
                rows.append((key, env, d))
    if not rows:
        return
    names = list(rows[0][2]["term_names"])
    fig, axes = plt.subplots(1, len(names), figsize=(2.3 * len(names), 4.2), constrained_layout=True, sharey=False)
    for ax, ti, n in zip(axes, range(len(names)), names):
        labels, vals, cols, hatches = [], [], [], []
        for key, env, d in rows:
            labels.append(f"{'평지' if env == 'flat' else '처음'}")
            vals.append(d["term_rewards"][:, ti].mean())
            cols.append(POLICIES[key]["color"])
            hatches.append(None if env == "flat" else "//")
        bars = ax.bar(range(len(vals)), vals, color=cols, width=0.8)
        for b, hch in zip(bars, hatches):
            if hch:
                b.set_hatch(hch)
                b.set_edgecolor("#fcfcfb")
        ax.set_xticks(range(len(vals)), labels, fontsize=8)
        ax.set_title(n, fontsize=10)
        ax.grid(axis="x", visible=False)
        ax.axhline(0, color="#8a8984", lw=0.8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=POLICIES[k]["color"]) for k in MAIN if any(r[0] == k for r in rows)]
    fig.legend(handles, [POLICIES[k]["label"] for k in MAIN if any(r[0] == k for r in rows)], loc="upper right",
               ncol=3, fontsize=9)
    fig.suptitle("보상 항목별 에피소드 합 (빗금 = 처음 보는 환경)", x=0.01, ha="left", fontsize=12, y=1.06)
    save(fig, "fig6_reward_terms.png")


def plot_friction_sweep():
    sweep = os.path.join(R, "sweep")
    if not os.path.isdir(sweep):
        return
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.axvspan(0.4, 1.2, color="#e6e5e0", alpha=0.6, lw=0)
    ax.text(0.7, 3, "기존 학습 범위 0.4~1.2", ha="center", fontsize=8, color=TEXT2)
    ax.annotate("", xy=(2.0, 7), xytext=(0.1, 7), arrowprops=dict(arrowstyle="<->", color=TEXT2, lw=0.8))
    ax.text(0.45, 8.2, "넓은 범위 학습 0.1~2.0", ha="center", fontsize=8, color=TEXT2)
    ends = []
    for key in ["baseline", "scan_warm", "teacher", "student", "student_wide", "student_imu"]:
        fs, ms, ses = [], [], []
        for f in ["0.1", "0.2", "0.4", "0.6", "0.8", "1.0", "1.2", "1.5", "2.0"]:
            path = os.path.join(sweep, f"{key}_f{f}.npz")
            if os.path.exists(path):
                r = np.load(path, allow_pickle=True)["reward_total"]
                fs.append(float(f)); ms.append(r.mean()); ses.append(r.std() / np.sqrt(len(r)))
        if not fs:
            continue
        p = POLICIES.get(key) or VARIANTS[key]
        ms, ses = np.array(ms), np.array(ses)
        ax.fill_between(fs, ms - ses, ms + ses, color=p["color"], alpha=0.12, lw=0)
        ax.plot(fs, ms, marker="o", ls=p.get("ls", "-"), color=p["color"], lw=2, ms=6, mec="#fcfcfb", mew=1.5, label=p["label"])
        ends.append([ms[-1], p["short"], fs[-1]])
    # direct end labels, pushed apart so they do not overlap
    ends.sort(key=lambda e: -e[0])
    for i in range(1, len(ends)):
        ends[i][0] = min(ends[i][0], ends[i - 1][0] - 2.8)
    for y, label, x in ends:
        ax.text(x * 1.05, y, label, va="center", fontsize=8, color=TEXT2)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0], ["0.1", "0.2", "0.4", "0.6", "0.8", "1.0", "1.2", "1.5", "2.0"])
    ax.minorticks_off()
    ax.set_xlim(0.09, 3.2)
    ax.set_ylim(0, None)
    ax.set_xlabel("발-지면 마찰 계수 (로그 축, 처음 보는 지형, 모든 접촉면 동일)")
    ax.set_ylabel("에피소드 누적 보상 (평균 ± 표준오차)")
    ax.set_title("마찰 스윕 — 처음 보는 지형에서 바닥 물성이 바뀔 때", loc="left", fontsize=12)
    ax.legend(loc="lower left", fontsize=8)
    save(fig, "fig7_friction_sweep.png")


def summary_table():
    lines = ["| 정책 | 환경 | reward mean | std | steps mean | 전진 거리 [m] | 넘어짐 비율 |", "|---|---|---|---|---|---|---|"]
    for key in POLICIES:
        for env in ENVS:
            d = load_eval(key, env)
            if d is None:
                continue
            dist = d["end_pos"][:, 0] - d["start_pos"][:, 0]
            fell = 1.0 - d["timed_out"].mean()
            lines.append(
                f"| {POLICIES[key]['label']} | {ENVS[env]} | {d['reward_total'].mean():.2f} | {d['reward_total'].std():.2f} |"
                f" {d['steps'].mean():.1f} | {dist.mean():.1f} | {fell * 100:.0f}% |"
            )
    text = "\n".join(lines)
    with open(os.path.join(R, "summary.md"), "w") as f:
        f.write(text + "\n")
    print(text)


plot_terrains()
plot_training()
plot_eval_bars()
plot_eval_by_terrain()
plot_trajectories()
plot_terms()
plot_friction_sweep()
summary_table()
