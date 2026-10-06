#!/usr/bin/env bash
# Seed repeats of the teacher-student pipeline (the IMU condition was removed after the 2026-10-05 ablation, see
# results/README.md section 12).
#   usage: run_seed_repeats.sh "<seeds>"      e.g. run_seed_repeats.sh "1 2"
# For every (condition, seed): teacher warm-started from v4 (2000 iter) -> student DAgger (1500 iter) -> evaluation
# Results: results/seeds/<cond>_s<seed>_<teacher|student>_<flat|test>.npz
set -u
cd "$(dirname "$0")/../.."
SEEDS=${1:-"1 2"}
V4=$(ls -d logs/rsl_rl/ant_rough/*_v4_warm | tail -1)/model_1999.pt
BASE=logs/rsl_rl/ant/2026-09-28_14-21-51_ant_baseline/model_999.pt
mkdir -p results/seeds

eval_run() { timeout -s KILL 300 ./isaaclab.sh -p scripts/analysis/evaluate_ant.py --seed 24 --num_envs 100 --headless "$@" 2>&1 | grep "\[RESULT\]"; }

for seed in $SEEDS; do
  for cond in rma; do
    TASK=Isaac-Ant-RMA; NEW=323
    tag=${cond}_s${seed}
    echo "===== $tag ($(date +%H:%M))"
    python scripts/analysis/make_warmstart_checkpoint.py --src "$V4" --dst logs/rsl_rl/ant_rma/${tag}_teacher_init/model_0.pt \
        --new_inputs $NEW --std 0.2 > /dev/null
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task $TASK-v0 --headless --num_envs 2048 --seed $seed \
        --max_iterations 2000 --run_name ${tag}_teacher --resume --load_run ${tag}_teacher_init --checkpoint model_0.pt \
        > /dev/null 2>&1
    TEACHER=$(ls -d logs/rsl_rl/ant_rma/*_${tag}_teacher | tail -1)/model_1999.pt
    python scripts/analysis/make_student_checkpoints.py init --teacher "$TEACHER" \
        --dst logs/rsl_rl/ant_rma/${tag}_student_init/model_0.pt > /dev/null
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task $TASK-v0 --agent rsl_rl_distillation_cfg_entry_point \
        --headless --num_envs 2048 --seed $seed --max_iterations 1500 --run_name ${tag}_student \
        --load_run ${tag}_student_init --checkpoint model_0.pt > /dev/null 2>&1
    STUDENT=$(ls -d logs/rsl_rl/ant_rma/*_${tag}_student | tail -1)/model_1499.pt
    python scripts/analysis/make_student_checkpoints.py export --student "$STUDENT" --flat "$BASE" \
        --dst_dir logs/rsl_rl/ant_rma/${tag} > /dev/null
    for env in Flat Test; do
      e=$(echo $env | tr A-Z a-z)
      echo -n "$tag teacher $e: "; eval_run --task $TASK-$env-v0 --checkpoint "$TEACHER" --out results/seeds/${tag}_teacher_$e.npz
      echo -n "$tag student $e: "; eval_run --task $TASK-$env-v0 --agent rsl_rl_student_cfg_entry_point \
          --checkpoint logs/rsl_rl/ant_rma/${tag}/student_plain/model_0.pt --out results/seeds/${tag}_student_$e.npz
    done
  done
done
echo "SEED REPEATS DONE ($(date +%H:%M))"
