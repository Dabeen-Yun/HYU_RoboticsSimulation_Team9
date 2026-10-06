"""Slide-ready figures: Baseline vs Ours (teacher-student) only.

Reads the per-robot evaluation files written by evaluate_ant.py and writes PNG (300 dpi) + PDF (vector) figures sized
for a 16:9 slide, once with English and once with Korean labels.

    python scripts/analysis/plot_presentation.py            # -> results/figures/presentation/{en,ko}/
    python scripts/analysis/plot_presentation.py --lang en

Re-run after more seed repeats land in results/seeds/ (rma_s<k>_{teacher,student}_{flat,test}.npz); fig_seeds picks them up.
"""

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("--results", default="results")
parser.add_argument("--lang", default="both", choices=["en", "ko", "both"])
args = parser.parse_args()
R = args.results

# ---------------------------------------------------------------- style (reference palette, light mode)
for path in glob.glob("/usr/share/fonts/opentype/noto/NotoSansCJK-*.ttc"):
    font_manager.fontManager.addfont(path)
SURFACE, INK, INK2, MUTED, GRID = "#ffffff", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
plt.rcParams.update(
    {
        "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "axes.unicode_minus": False,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "text.color": INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
)
# categorical slots 1-3 (validated all-pairs); colour follows the policy everywhere
COLOR = {"baseline": "#eb6834", "student": "#2a78d6", "teacher": "#1baf7a"}
ORDER = ["baseline", "teacher", "student"]
ORACLE = 67.2  # policy trained directly on the unseen env (results/diag/oracle_test.npz): empirical ceiling on mesh ground

TXT = {
    "en": {
        "baseline": "Baseline",
        "teacher": "Teacher†",
        "student": "Ours (student)",
        "teacher_note": "† privileged physics input; not deployable",
        "flat": "Original env\n(Isaac-Ant-v0)",
        "test": "Unseen env\n(held out)",
        "flat1": "Original env (Isaac-Ant-v0)",
        "test1": "Unseen env (held out)",
        "reward": "Episode reward (original Ant reward)",
        "oracle": f"Oracle ceiling {ORACLE:.1f}\n(trained on the unseen env)",
        "motiv_title": "Baseline breaks as soon as the ground changes",
        "m_plane": "Original env\n(analytic ground plane)",
        "m_mesh": "Same flat ground,\ntriangle-mesh collider",
        "m_test": "Unseen terrain\n+ unseen physics",
        "m_note": "Same policy, same reward, same seed (24), 100 robots each. Error bars: 95% CI of the mean.",
        "plane": "Original\nflat plane",
        "wave": "Waves\n(unseen)",
        "discrete_obstacles": "Obstacles\n(unseen)",
        "random_rough_hard": "Rough ×1.7\n(beyond train)",
        "terrain_title": "Where does the gain come from? (per terrain type)",
        "friction": "Ground friction coefficient (all contacts, unseen terrain)",
        "train_range": "training range",
        "test_range": "unseen env",
        "fric_title": "Robustness to ground friction",
        "time": "Time [s]",
        "alive": "Robots still walking [%]",
        "progress": "Forward progress [m]",
        "dyn_a": "(a) Survival",
        "dyn_b": "(b) Distance covered (fallen robots keep their last position)",
        "terms": {
            "progress": "progress",
            "alive": "alive",
            "upright": "upright",
            "move_to_target": "move to target",
            "action_l2": "action L2",
            "energy": "energy",
            "joint_pos_limits": "joint limits",
        },
        "term_x": "Mean reward per episode",
        "seed": "Training seed",
        "seed_title": "Reproducibility across training seeds",
        "baseline_band": "Baseline (mean ± 95% CI)",
        "delta": "{d:+.1f} ({pct:+.0f}%)",
        "pval": "p = {p}",
        "n_note": "n = {n} robots",
        # method diagram
        "d_obs": "Deployable observations (653-D)",
        "d_prop": "Proprioception 60",
        "d_scan": "Height scan 273\n(RayCaster 2.0 × 1.2 m)",
        "d_contact": "Foot contacts 16\n(normal, tangential, μ-usage, slip)",
        "d_hist": "8-step history 304\n(joint vel, action, base vel, contacts)",
        "d_priv": "Privileged physics 3\n(friction, mass ratio, plane flag)",
        "d_teacher": "Teacher π_T\nMLP 400-200-100",
        "d_student": "Student π_S\nMLP 400-200-100",
        "d_ppo": "Stage 1: PPO, original reward\n2000 iters (warm-start)",
        "d_dagger": "Stage 2: DAgger distillation\nmin ‖π_S(o) − π_T(o, e)‖²   1500 iters",
        "d_deploy": "Deployed policy: student only\n(no simulator-only inputs)",
        "d_act": "Joint torques 8",
        "d_title": "Teacher-student training: infer the ground from how it feels",
    },
    "ko": {
        "baseline": "Baseline",
        "teacher": "교사†",
        "student": "Ours (학생)",
        "teacher_note": "† 정답 물성을 입력으로 받음 (배포 불가)",
        "flat": "기존 환경\n(Isaac-Ant-v0)",
        "test": "처음 보는 환경\n(학습에 미사용)",
        "flat1": "기존 환경 (Isaac-Ant-v0)",
        "test1": "처음 보는 환경 (학습에 미사용)",
        "reward": "에피소드 보상 (원본 Ant 보상)",
        "oracle": f"오라클 상한 {ORACLE:.1f}\n(처음 보는 환경에서 직접 학습)",
        "motiv_title": "Baseline은 바닥이 바뀌는 순간 무너진다",
        "m_plane": "기존 환경\n(해석적 평면 바닥)",
        "m_mesh": "같은 평지,\n삼각형 메쉬 충돌체",
        "m_test": "처음 보는 지형\n+ 처음 보는 물성",
        "m_note": "같은 정책, 같은 보상, 같은 seed(24), 각 100개체. 오차 막대: 평균의 95% 신뢰구간.",
        "plane": "기존\n평면 바닥",
        "wave": "파도\n(미학습)",
        "discrete_obstacles": "장애물\n(미학습)",
        "random_rough_hard": "거친 요철 ×1.7\n(학습 범위 밖)",
        "terrain_title": "어디서 성능 차이가 나는가? (지형별)",
        "friction": "바닥 마찰 계수 (모든 접촉면, 처음 보는 지형)",
        "train_range": "학습 범위",
        "test_range": "평가 환경",
        "fric_title": "바닥 마찰 변화에 대한 강건성",
        "time": "시간 [s]",
        "alive": "넘어지지 않은 개체 [%]",
        "progress": "전진 거리 [m]",
        "dyn_a": "(a) 생존율",
        "dyn_b": "(b) 전진 거리 (넘어진 개체는 마지막 위치 유지)",
        "terms": {
            "progress": "전진(progress)",
            "alive": "생존(alive)",
            "upright": "자세(upright)",
            "move_to_target": "목표 방향",
            "action_l2": "행동 크기",
            "energy": "에너지",
            "joint_pos_limits": "관절 한계",
        },
        "term_x": "에피소드당 평균 보상",
        "seed": "학습 seed",
        "seed_title": "학습 seed에 따른 재현성",
        "baseline_band": "Baseline (평균 ± 95% CI)",
        "delta": "{d:+.1f} ({pct:+.0f}%)",
        "pval": "p = {p}",
        "n_note": "n = {n}",
        "d_obs": "배포 가능한 관측 (653차원)",
        "d_prop": "고유 감각 60",
        "d_scan": "높이맵 273\n(RayCaster 2.0 × 1.2 m)",
        "d_contact": "발 접촉 16\n(수직력, 접선력, 마찰 사용량, 미끄러짐)",
        "d_hist": "최근 8스텝 이력 304\n(관절 속도, 행동, 몸통 속도, 접촉)",
        "d_priv": "정답 물성 3\n(마찰, 질량 비율, 평면 여부)",
        "d_teacher": "교사 π_T\nMLP 400-200-100",
        "d_student": "학생 π_S\nMLP 400-200-100",
        "d_ppo": "1단계: PPO, 원본 보상\n2000 iter (이어 학습)",
        "d_dagger": "2단계: DAgger 증류\nmin ‖π_S(o) − π_T(o, e)‖²   1500 iter",
        "d_deploy": "배포 정책: 학생만 사용\n(시뮬레이터 전용 입력 없음)",
        "d_act": "관절 토크 8",
        "d_title": "교사-학생 학습: 발의 감각으로 바닥을 추정한다",
    },
}


# ---------------------------------------------------------------- data
def load(path):
    path = os.path.join(R, path)
    return dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else None


def ev(policy, env):
    return load(f"eval/{policy}_{env}.npz")


def ci95(x):
    x = np.asarray(x, float)
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def pstr(a, b):
    p = stats.ttest_ind(a, b, equal_var=False).pvalue
    if p < 1e-3:
        e = int(np.floor(np.log10(p)))
        return f"{p / 10**e:.0f}×10$^{{{e}}}$"
    return f"{p:.2f}" if p >= 0.01 else f"{p:.3f}"


def forward(d):
    return d["traj"][:, :, 0] - d["traj"][:, :1, 0]


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), facecolor=SURFACE)
    plt.close(fig)
    print("[saved]", os.path.join(out, name + ".png"))


def bar_with_dots(ax, x, vals, color, width, label=None, dots=True, seed=0):
    m, c = np.mean(vals), ci95(vals)
    ax.bar(x, m, width, color=color, edgecolor=SURFACE, linewidth=2, label=label, zorder=2)
    if dots:
        rng = np.random.default_rng(seed)
        jit = rng.uniform(-width * 0.32, width * 0.32, len(vals))
        ax.scatter(x + jit, vals, s=9, color=INK, alpha=0.18, linewidths=0, zorder=3)
    ax.errorbar(x, m, yerr=c, fmt="none", ecolor=INK, elinewidth=1.6, capsize=5, capthick=1.6, zorder=4)
    return m, c


# ---------------------------------------------------------------- figures
def fig_motivation(T, out):
    """Intro: the baseline's score is tied to the analytic ground plane, not to 'flatness'."""
    rows = [("m_plane", ev("baseline", "flat")), ("m_mesh", ev("baseline", "flatmesh")), ("m_test", ev("baseline", "test"))]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ref = rows[0][1]["reward_total"].mean()
    for i, (k, d) in enumerate(rows):
        r = d["reward_total"]
        m, c = r.mean(), ci95(r)
        ax.barh(i, m, 0.62, color=COLOR["baseline"], alpha=1.0 if i == 0 else 0.55, edgecolor=SURFACE, linewidth=2)
        ax.errorbar(m, i, xerr=c, fmt="none", ecolor=INK, elinewidth=1.6, capsize=5)
        lab = f"{m:.1f}" if i == 0 else f"{m:.1f}   ({(m / ref - 1) * 100:+.0f}%)".replace("-", "−")
        ax.text(m + c + 3, i, lab, va="center", fontsize=15, color=INK, fontweight="bold" if i else "normal")
    ax.set_yticks(range(len(rows)), [T[k] for k, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 175)
    ax.set_xlabel(T["reward"])
    ax.set_title(T["motiv_title"], loc="left", fontweight="bold", pad=12)
    fig.text(0.01, -0.09, T["m_note"], fontsize=11, color=MUTED, ha="left")
    save(fig, out, "fig1_motivation_baseline_collapse")


def fig_headline(T, out):
    """Main result: two environments x {Baseline, Teacher, Student}."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, env in zip(axes, ["flat", "test"]):
        data = {p: ev(p, env)["reward_total"] for p in ORDER}
        for i, p in enumerate(ORDER):
            m, c = bar_with_dots(ax, i, data[p], COLOR[p], 0.66, seed=i)
            ax.text(i, 4, f"{m:.1f}", ha="center", va="bottom", fontsize=17, color=SURFACE, fontweight="bold", zorder=5)
        # Ours vs Baseline bracket
        b, s = data["baseline"], data["student"]
        top = max(b.max(), s.max()) * 0 + max(np.percentile(b, 99), np.percentile(s, 99)) + 14
        top = min(top, 222)
        ax.plot([0, 0, 2, 2], [top - 4, top, top, top - 4], color=INK2, lw=1.2)
        d = s.mean() - b.mean()
        ax.text(1, top + 3, T["delta"].format(d=d, pct=d / b.mean() * 100) + "   " + T["pval"].format(p=pstr(s, b)),
                ha="center", fontsize=14, color=INK, fontweight="bold")
        if env == "test":
            ax.axhline(ORACLE, color=INK2, lw=1.4, ls=(0, (5, 3)), zorder=1)
            ax.text(2.64, ORACLE, T["oracle"], fontsize=11, color=INK2, ha="left", va="center", clip_on=False)
        ax.set_xticks(range(3), [T[p] for p in ORDER])
        for lbl, p in zip(ax.get_xticklabels(), ORDER):
            lbl.set_fontweight("bold" if p == "student" else "normal")
        ax.set_title(T[env + "1"], fontweight="bold", pad=10)
        ax.set_xlim(-0.6, 2.6)
    axes[0].set_ylabel(T["reward"])
    axes[0].set_ylim(0, 245)
    axes[1].tick_params(axis="y", left=False)
    fig.text(0.01, -0.03, T["teacher_note"] + "   ·   " + T["m_note"].split(". ", 1)[1] + "  " + T["n_note"].format(n=100),
             fontsize=11, color=MUTED, ha="left")
    save(fig, out, "fig2_main_result")


def fig_terrain(T, out):
    """Per terrain type: original flat plane + the three column types of the unseen env."""
    groups = [("plane", None)] + [(t, t) for t in ["wave", "discrete_obstacles", "random_rough_hard"]]
    fig, ax = plt.subplots(figsize=(12, 5.4))
    w = 0.26
    for p_i, p in enumerate(ORDER):
        for g_i, (g, tname) in enumerate(groups):
            if tname is None:
                vals = ev(p, "flat")["reward_total"]
            else:
                d = ev(p, "test")
                vals = d["reward_total"][d["terrain_type_names"][d["terrain_col"]] == tname]
            x = g_i + (p_i - 1) * (w + 0.02)
            m, c = bar_with_dots(ax, x, vals, COLOR[p], w, label=T[p] if g_i == 0 else None, seed=g_i * 3 + p_i)
            ax.text(x, 3, f"{m:.0f}", ha="center", va="bottom", fontsize=13, color=SURFACE, fontweight="bold", zorder=5)
    # Ours-vs-baseline delta per group
    for g_i, (g, tname) in enumerate(groups):
        def sel(p):
            if tname is None:
                return ev(p, "flat")["reward_total"]
            d = ev(p, "test")
            return d["reward_total"][d["terrain_type_names"][d["terrain_col"]] == tname]
        b, s = sel("baseline"), sel("student")
        ax.text(g_i, 222, f"Δ {s.mean() - b.mean():+.1f}\np = {pstr(s, b)}", ha="center", fontsize=12, color=INK2, va="top")
    ax.axvline(0.5, color=MUTED, lw=1, ls=":")
    ax.set_xticks(range(len(groups)), [T[g] for g, _ in groups])
    ax.set_ylabel(T["reward"])
    ax.set_ylim(0, 225)
    ax.set_title(T["terrain_title"], loc="left", fontweight="bold", pad=40)
    ax.legend(loc="lower left", ncol=3, bbox_to_anchor=(0.0, 1.0), borderaxespad=0.2)
    n = np.bincount(ev("student", "test")["terrain_col"]).tolist()
    fig.text(0.01, -0.03, T["teacher_note"] + "   ·   Δ = Ours − Baseline (Welch t-test)   ·   n = 100 / 40 / 30 / 30",
             fontsize=11, color=MUTED, ha="left")
    save(fig, out, "fig3_per_terrain")


def fig_friction(T, out):
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.axvspan(0.4, 1.2, color="#f0efeb", zorder=0)
    ax.text(0.693, 3, T["train_range"] + " (0.4–1.2)", ha="center", fontsize=12, color=INK2)
    ax.annotate("", xy=(0.3, 99), xytext=(0.7, 99), arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.2))
    ax.text(0.458, 101, T["test_range"] + " (0.3–0.7)", ha="center", fontsize=12, color=INK2)
    for p in ORDER:
        fs, ms, cs = [], [], []
        for f in sorted(glob.glob(os.path.join(R, "sweep", f"{p}_f*.npz"))):
            mf = re.search(r"_f([0-9.]+)\.npz$", f)
            r = np.load(f)["reward_total"]
            fs.append(float(mf.group(1)))
            ms.append(r.mean())
            cs.append(ci95(r))
        fs, ms, cs = map(np.array, (fs, ms, cs))
        o = np.argsort(fs)
        fs, ms, cs = fs[o], ms[o], cs[o]
        ax.fill_between(fs, ms - cs, ms + cs, color=COLOR[p], alpha=0.14, linewidth=0)
        ax.plot(fs, ms, color=COLOR[p], lw=2.4, marker="o", ms=8, mec=SURFACE, mew=2, label=T[p],
                zorder=3 if p == "student" else 2)
    ax.set_xscale("log")
    ticks = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(0.09, 2.3)
    ax.set_ylim(0, 108)
    ax.set_xlabel(T["friction"])
    ax.set_ylabel(T["reward"])
    ax.set_title(T["fric_title"], loc="left", fontweight="bold", pad=12)
    ax.legend(loc="upper right")
    fig.text(0.01, -0.03, T["teacher_note"] + "   ·   " + T["m_note"].split(". ", 1)[1].replace("오차 막대", "음영")
             .replace("Error bars", "Bands"), fontsize=11, color=MUTED, ha="left")
    save(fig, out, "fig4_friction_sweep")


def fig_dynamics(T, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={"wspace": 0.22})
    for p in ORDER:
        d = ev(p, "test")
        dt = float(d["dt"])
        nT = d["traj"].shape[1]
        t = np.arange(nT) * dt
        steps = d["steps"]
        fallen = ~d["timed_out"]
        alive = 100 * np.array([(~(fallen & (steps <= k))).mean() for k in range(nT)])
        axes[0].plot(t, alive, color=COLOR[p], lw=2.4, label=T[p], drawstyle="steps-post")
        fx = forward(d)
        m = fx.mean(0)
        c = stats.t.ppf(0.975, fx.shape[0] - 1) * fx.std(0, ddof=1) / np.sqrt(fx.shape[0])
        axes[1].fill_between(t, m - c, m + c, color=COLOR[p], alpha=0.14, linewidth=0)
        axes[1].plot(t, m, color=COLOR[p], lw=2.4, label=T[p])
        axes[0].text(t[-1] + 0.2, alive[-1], f"{alive[-1]:.0f}%", va="center", fontsize=13)
        axes[1].text(t[-1] + 0.2, m[-1], f"{m[-1]:.0f} m", va="center", fontsize=13,
                     fontweight="bold" if p == "student" else "normal")
    axes[0].set_ylim(0, 104)
    axes[0].set_ylabel(T["alive"])
    axes[1].set_ylabel(T["progress"])
    axes[1].set_ylim(bottom=0)
    for ax, k in zip(axes, ["dyn_a", "dyn_b"]):
        ax.set_xlabel(T["time"])
        ax.set_xlim(0, 18)
        ax.set_title(T[k], loc="left", fontsize=14, pad=8)
        ax.grid(axis="x", visible=True)
    axes[1].legend(loc="upper left")
    fig.suptitle(T["test1"], x=0.01, ha="left", fontweight="bold", fontsize=16, y=1.03)
    fig.text(0.01, -0.04, T["teacher_note"] + "   ·   n = 100", fontsize=11, color=MUTED, ha="left")
    save(fig, out, "fig5_survival_progress_unseen")


def fig_terms(T, out):
    names = list(ev("baseline", "test")["term_names"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True, gridspec_kw={"wspace": 0.06})
    pols = ["baseline", "student"]
    h = 0.36
    for ax, env in zip(axes, ["flat", "test"]):
        for j, p in enumerate(pols):
            d = ev(p, env)
            tr = d["term_rewards"]
            m = tr.mean(0)
            c = np.array([ci95(tr[:, k]) for k in range(tr.shape[1])])
            y = np.arange(len(names)) + (j - 0.5) * (h + 0.04)
            ax.barh(y, m, h, color=COLOR[p], edgecolor=SURFACE, linewidth=2, label=T[p])
            ax.errorbar(m, y, xerr=c, fmt="none", ecolor=INK, elinewidth=1.2, capsize=3)
            for yy, mm, cc in zip(y, m, c):
                ax.text(mm + np.sign(mm or 1) * (cc + 2), yy, f"{mm:.1f}", va="center",
                        ha="left" if mm >= 0 else "right", fontsize=11, color=INK)
        ax.axvline(0, color=INK2, lw=1)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.set_title(T[env + "1"], fontweight="bold", pad=10)
        ax.set_xlabel(T["term_x"])
        ax.set_xlim(-45 if env == "flat" else -30, 175 if env == "flat" else 100)
    axes[0].set_yticks(range(len(names)), [T["terms"].get(n, n) for n in names])
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right")
    save(fig, out, "fig6_reward_terms")


def fig_seeds(T, out):
    seeds = {0: {k: ev(k, e) for k in ("teacher", "student") for e in ("flat", "test")}}
    seeds[0] = {(k, e): ev(k, e) for k in ("teacher", "student") for e in ("flat", "test")}
    for f in sorted(glob.glob(os.path.join(R, "seeds", "rma_s*_student_flat.npz"))):
        s = int(re.search(r"rma_s(\d+)_", f).group(1))
        dd = {(k, e): load(f"seeds/rma_s{s}_{k}_{e}.npz") for k in ("teacher", "student") for e in ("flat", "test")}
        if all(v is not None for v in dd.values()):
            seeds[s] = dd
    S = sorted(seeds)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), sharey=False, gridspec_kw={"wspace": 0.18})
    for ax, env in zip(axes, ["flat", "test"]):
        b = ev("baseline", env)["reward_total"]
        ax.axhspan(b.mean() - ci95(b), b.mean() + ci95(b), color=COLOR["baseline"], alpha=0.15, linewidth=0)
        ax.axhline(b.mean(), color=COLOR["baseline"], lw=2, label=T["baseline_band"])
        for j, p in enumerate(["teacher", "student"]):
            xs = np.arange(len(S)) + (j - 0.5) * 0.22
            ms = np.array([seeds[s][(p, env)]["reward_total"].mean() for s in S])
            cs = np.array([ci95(seeds[s][(p, env)]["reward_total"]) for s in S])
            ax.errorbar(xs, ms, yerr=cs, fmt="o", color=COLOR[p], ms=10, mec=SURFACE, mew=2, elinewidth=1.8, capsize=4,
                        label=T[p])
            for x, m in zip(xs, ms):
                ax.text(x + 0.06, m, f"{m:.0f}", fontsize=11, va="center", ha="left")
        ax.set_xticks(range(len(S)), [str(s) for s in S])
        ax.set_xlim(-0.6, len(S) - 0.4)
        ax.set_xlabel(T["seed"])
        ax.set_title(T[env + "1"], fontweight="bold", pad=10)
    axes[0].set_ylim(80, 175)
    axes[1].set_ylim(0, 90)
    axes[0].set_ylabel(T["reward"])
    axes[1].legend(loc="lower right", fontsize=12)
    fig.suptitle(T["seed_title"], x=0.01, ha="left", fontweight="bold", fontsize=16, y=1.03)
    fig.text(0.01, -0.04, T["teacher_note"] + "   ·   " + T["m_note"].split(". ", 1)[1], fontsize=11, color=MUTED, ha="left")
    save(fig, out, "fig7_seed_repeats")


def fig_method(T, out):
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=12, bold=False, tc=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
                fontweight="bold" if bold else "normal", linespacing=1.3)

    def arrow(p0, p1, color=INK2, ls="-", lw=1.8):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=16, color=color, lw=lw, linestyle=ls,
                                     shrinkA=0, shrinkB=0))

    blue_bg, aqua_bg, grey_bg = "#e8f1fb", "#e4f6ef", "#f4f3f0"
    # privileged input (teacher only) on top, deployable observation stack below
    box(0.35, 5.3, 3.5, 0.8, T["d_priv"], SURFACE, COLOR["teacher"], fs=11.5)
    ax.add_patch(FancyBboxPatch((0.15, 0.12), 3.9, 4.95, boxstyle="round,pad=0.02,rounding_size=0.15", fc=grey_bg,
                                ec=MUTED, lw=1.2, ls=(0, (4, 3))))
    ax.text(2.1, 4.72, T["d_obs"], ha="center", fontsize=12.5, color=INK2, fontweight="bold")
    for y, k, new in zip([3.75, 2.68, 1.61, 0.54][::1], ["d_prop", "d_scan", "d_contact", "d_hist"], [False, True, True, True]):
        box(0.35, y - 0.3, 3.5, 0.82, T[k], SURFACE, COLOR["student"] if new else MUTED, fs=11.5)
    # networks
    box(5.7, 4.05, 2.6, 1.15, T["d_teacher"], aqua_bg, COLOR["teacher"], fs=13, bold=True)
    box(5.7, 1.75, 2.6, 1.15, T["d_student"], blue_bg, COLOR["student"], fs=13, bold=True)
    arrow((3.87, 5.7), (5.68, 4.95), color=COLOR["teacher"], ls=(0, (4, 3)))
    arrow((4.07, 3.0), (5.68, 4.35))
    arrow((4.07, 2.5), (5.68, 2.32))
    # outputs
    box(9.3, 4.25, 1.7, 0.75, T["d_act"], SURFACE, MUTED, fs=11.5)
    box(9.3, 1.95, 1.7, 0.75, T["d_act"], SURFACE, MUTED, fs=11.5)
    arrow((8.3, 4.62), (9.28, 4.62))
    arrow((8.3, 2.32), (9.28, 2.32))
    ax.text(7.0, 5.32, T["d_ppo"], ha="center", va="bottom", fontsize=11.5, color=INK2, linespacing=1.3)
    arrow((10.15, 4.22), (10.15, 2.73), color=COLOR["student"], lw=2.2)
    ax.text(10.35, 3.48, T["d_dagger"], ha="left", va="center", fontsize=11.5, color=INK, linespacing=1.35)
    box(5.4, 0.2, 5.6, 0.75, T["d_deploy"], blue_bg, COLOR["student"], fs=12, bold=True)
    arrow((7.0, 1.73), (7.0, 0.97), color=COLOR["student"])
    ax.set_title(T["d_title"], loc="left", fontweight="bold", fontsize=17, pad=6)
    save(fig, out, "fig0_method_teacher_student")


# ---------------------------------------------------------------- main
langs = ["en", "ko"] if args.lang == "both" else [args.lang]
for lang in langs:
    out = os.path.join(R, "figures", "presentation", lang)
    os.makedirs(out, exist_ok=True)
    T = TXT[lang]
    for f in (fig_method, fig_motivation, fig_headline, fig_terrain, fig_friction, fig_dynamics, fig_terms, fig_seeds):
        f(T, out)
