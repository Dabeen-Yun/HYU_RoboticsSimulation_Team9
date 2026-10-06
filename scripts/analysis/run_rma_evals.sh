#!/usr/bin/env bash
# Evaluation of the teacher-student (ground-property estimation) experiment.
#  1) original env ("flat") and held-out env ("test") with the protocol of play_one_episode.py (seed 24, 100 envs)
#  2) friction sweep on the held-out terrain: every collision shape gets the same friction f
# Results: results/eval/<policy>_<env>.npz and results/sweep/<policy>_f<f>.npz
set -u
cd "$(dirname "$0")/../.."

BASE=logs/rsl_rl/ant/2026-09-28_14-21-51_ant_baseline/model_999.pt
V4=$(ls -d logs/rsl_rl/ant_rough/*_v4_warm | tail -1)/model_1999.pt
TEACHER=$(ls -d logs/rsl_rl/ant_rma/*_teacher | tail -1)/model_1999.pt
STUDENT=logs/rsl_rl/ant_rma/student_plain/model_0.pt
STUDENT_GATED=logs/rsl_rl/ant_rma/student_gated/model_0.pt

run() { # out task agent ckpt [hydra overrides...]
    local out=$1 task=$2 agent=$3 ckpt=$4; shift 4
    timeout -s KILL 300 ./isaaclab.sh -p scripts/analysis/evaluate_ant.py --task "$task" --agent "$agent" --seed 24 \
        --num_envs 100 --headless --checkpoint "$ckpt" --out "$out" "$@" 2>&1 | grep "\[RESULT\]" | sed "s|^|$(basename $out .npz): |"
}

mkdir -p results/eval results/sweep results/official
run results/eval/teacher_flat.npz Isaac-Ant-RMA-Flat-v0 rsl_rl_cfg_entry_point "$TEACHER"
run results/eval/teacher_test.npz Isaac-Ant-RMA-Test-v0 rsl_rl_cfg_entry_point "$TEACHER"
run results/eval/student_flat.npz Isaac-Ant-RMA-Flat-v0 rsl_rl_student_cfg_entry_point "$STUDENT"
run results/eval/student_test.npz Isaac-Ant-RMA-Test-v0 rsl_rl_student_cfg_entry_point "$STUDENT"
run results/eval/student_gated_flat.npz Isaac-Ant-RMA-Flat-v0 rsl_rl_student_gated_cfg_entry_point "$STUDENT_GATED"
run results/eval/student_gated_test.npz Isaac-Ant-RMA-Test-v0 rsl_rl_student_gated_cfg_entry_point "$STUDENT_GATED"

for f in 0.2 0.4 0.6 0.8 1.0 1.2; do
    o="env.events.physics_material.params.static_friction_range=[$f,$f] env.events.physics_material.params.dynamic_friction_range=[$f,$f]"
    run results/sweep/baseline_f$f.npz Isaac-Ant-Test-Blind-v0 rsl_rl_cfg_entry_point "$BASE" $o
    run results/sweep/scan_warm_f$f.npz Isaac-Ant-Rough-Test-v0 rsl_rl_warm_cfg_entry_point "$V4" $o
    run results/sweep/teacher_f$f.npz Isaac-Ant-RMA-Test-v0 rsl_rl_cfg_entry_point "$TEACHER" $o
    run results/sweep/student_f$f.npz Isaac-Ant-RMA-Test-v0 rsl_rl_student_cfg_entry_point "$STUDENT" $o
done

# official numbers for the final candidate (unmodified evaluation script)
for line in "student_gated|flat|Isaac-Ant-RMA-Flat-v0" "student_gated|test|Isaac-Ant-RMA-Test-v0"; do
    IFS='|' read -r policy env task <<< "$line"
    timeout -s KILL 300 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py --task "$task" \
        --agent rsl_rl_student_gated_cfg_entry_point --seed 24 --num_envs 100 --headless --checkpoint "$STUDENT_GATED" 2>&1 \
        | grep -E "\[RESULT\]|Completed first" | tee results/official/${policy}_${env}.txt
done
