#!/usr/bin/env bash
# [ant_robust_bundle] Evaluate the baseline, our final policy and the 9 bundle checkpoints (all 4096 envs x 1000 iter,
# seed 42) in the original env and the 8 bundle test envs. Protocol of play_one_episode.py: seed 24, 100 envs.
# Results: results/bundle/<policy>__<env>.npz
set -u
cd "$(dirname "$0")/../.."
mkdir -p results/bundle
ENVS="Original LowFriction Heavy WeakMotor Rough Push Heading Combined TM"
ev() { timeout -s KILL 300 ./isaaclab.sh -p scripts/analysis/evaluate_ant.py --seed 24 --num_envs 100 --headless "$@" 2>&1 | grep "\[RESULT\]"; }
blind_task() { [ $1 = Original ] && echo Isaac-Ant-v0 || echo Isaac-Ant-Test-$1-v0; }
ours_task() { [ $1 = Original ] && echo Isaac-Ant-RMA-Flat-v0 || echo Isaac-Ant-RMA-Bundle-$1-v0; }

for env in $ENVS; do
    echo -n "ours $env: "; ev --task $(ours_task $env) --agent rsl_rl_finetune_cfg_entry_point --checkpoint checkpoints/ant_final.pt \
        --out results/bundle/ours__$env.npz
    echo -n "baseline $env: "; ev --task $(blind_task $env) --checkpoint checkpoints/ant_baseline.pt --out results/bundle/baseline__$env.npz
    for cond in dr dr_rough dr_rough_pushnoise dr_rough_smooth dr_rough_pushnoise_smooth dr_tm dr_tm_pushnoise dr_tm_smooth dr_tm_pushnoise_smooth; do
        echo -n "$cond $env: "; ev --task $(blind_task $env) --checkpoint checkpoints/ant_robust_bundle/$cond/model_999.pt \
            --out results/bundle/${cond}__$env.npz
    done
done
echo "BUNDLE EVALS DONE"
