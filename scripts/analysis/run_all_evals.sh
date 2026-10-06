#!/usr/bin/env bash
# Evaluate every policy of the Ant generalization study on the original environment ("flat") and on the held-out
# environment ("test"), with the protocol of play_one_episode.py (seed 24, 100 envs, first episode).
# Results: results/eval/<policy>_<env>.npz, official play_one_episode.py output in results/official/.
set -u
cd "$(dirname "$0")/../.."

BASE=logs/rsl_rl/ant/2026-09-28_14-21-51_ant_baseline/model_999.pt
V3=$(ls -d logs/rsl_rl/ant_rough/*_v3 | tail -1)/model_2999.pt
V4=$(ls -d logs/rsl_rl/ant_rough/*_v4_warm | tail -1)/model_1999.pt
V5=$(ls -d logs/rsl_rl/ant_rough/*_v5_warm_rehearsal | tail -1)/model_1999.pt
BLIND=$(ls -d logs/rsl_rl/ant_rough_blind/*_blind_warm | tail -1)/model_1999.pt
OURS=logs/rsl_rl/ant_rough/gated_v4/model_0.pt

mkdir -p results/eval results/official
# policy | env | task | agent entry point | checkpoint
RUNS="
baseline|flat|Isaac-Ant-v0|rsl_rl_cfg_entry_point|$BASE
baseline|test|Isaac-Ant-Test-Blind-v0|rsl_rl_cfg_entry_point|$BASE
baseline|flatmesh|Isaac-Ant-Blind-Flat-v0|rsl_rl_cfg_entry_point|$BASE
blind_warm|flat|Isaac-Ant-v0|rsl_rl_cfg_entry_point|$BLIND
blind_warm|test|Isaac-Ant-Test-Blind-v0|rsl_rl_cfg_entry_point|$BLIND
scan_scratch|flat|Isaac-Ant-Rough-Flat-v0|rsl_rl_cfg_entry_point|$V3
scan_scratch|test|Isaac-Ant-Rough-Test-v0|rsl_rl_cfg_entry_point|$V3
scan_warm|flat|Isaac-Ant-Rough-Flat-v0|rsl_rl_warm_cfg_entry_point|$V4
scan_warm|test|Isaac-Ant-Rough-Test-v0|rsl_rl_warm_cfg_entry_point|$V4
scan_rehearsal|flat|Isaac-Ant-Rough-Flat-v0|rsl_rl_warm_cfg_entry_point|$V5
scan_rehearsal|test|Isaac-Ant-Rough-Test-v0|rsl_rl_warm_cfg_entry_point|$V5
gated_v4|flat|Isaac-Ant-Rough-Flat-v0|rsl_rl_gated_cfg_entry_point|$OURS
gated_v4|test|Isaac-Ant-Rough-Test-v0|rsl_rl_gated_cfg_entry_point|$OURS
"
for line in $RUNS; do
    IFS='|' read -r policy env task agent ckpt <<< "$line"
    out=results/eval/${policy}_${env}.npz
    echo "== $policy / $env ($task)"
    timeout -s KILL 300 ./isaaclab.sh -p scripts/analysis/evaluate_ant.py --task "$task" --agent "$agent" --seed 24 \
        --num_envs 100 --headless --checkpoint "$ckpt" --out "$out" --label "$policy" 2>&1 | grep "\[RESULT\]"
done

# official numbers for the report (unmodified evaluation script)
for line in "gated_v4|flat|Isaac-Ant-Rough-Flat-v0|rsl_rl_gated_cfg_entry_point|$OURS" \
            "gated_v4|test|Isaac-Ant-Rough-Test-v0|rsl_rl_gated_cfg_entry_point|$OURS" \
            "baseline|flat|Isaac-Ant-v0|rsl_rl_cfg_entry_point|$BASE" \
            "baseline|test|Isaac-Ant-Test-Blind-v0|rsl_rl_cfg_entry_point|$BASE"; do
    IFS='|' read -r policy env task agent ckpt <<< "$line"
    echo "== official: $policy / $env ($task)"
    timeout -s KILL 300 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py --task "$task" \
        --agent "$agent" --seed 24 --num_envs 100 --headless --checkpoint "$ckpt" 2>&1 \
        | grep -E "\[RESULT\]|Completed first" | tee results/official/${policy}_${env}.txt
done
