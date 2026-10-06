#!/usr/bin/env bash
# Train the final policy (checkpoints/ant_final.pt) from the lecture baseline.
#
#   baseline (Isaac-Ant-v0)  ->  1. height-scan terrain expert  ->  2. teacher (privileged ground properties)
#                            ->  3. student (distillation, deployable sensors only)  ->  4. conservative PPO fine-tuning
#
# Every stage: 4096 envs x 1000 iterations, training seed 42 (same as the lecture baseline). The final checkpoint is the
# last iteration of stage 4 (fixed in advance, not selected on the test environment).
#
#   usage: ./scripts/final/train_final.sh            (about 1.5 h on an RTX 3070 Ti)
set -eu
cd "$(dirname "$0")/../.."

NUM_ENVS=${NUM_ENVS:-4096}
ITERS=${ITERS:-1000}
SEED=${SEED:-42}
TAG=${TAG:-final}
BASELINE=checkpoints/ant_baseline.pt
LOG=results/final
mkdir -p $LOG

train() { ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --headless --num_envs $NUM_ENVS --seed $SEED \
              --max_iterations $ITERS "$@"; }
# newest run "<date>_<name>" in an experiment folder -> its last checkpoint (empty if there is none)
last() { local run; run=$(ls -d "$1"/*_"$2" 2>/dev/null | tail -1); [ -n "$run" ] && ls -v "$run"/model_*.pt | tail -1 || true; }
# a stage is done when its run reached the last iteration; then it is skipped (lets an interrupted pipeline resume)
done_stage() { case "$(last "$1" "$2")" in *model_$((ITERS - 1)).pt) return 0 ;; *) return 1 ;; esac; }

echo "== 1/4 terrain expert, warm-started from the baseline ($(date +%H:%M))"
if ! done_stage logs/rsl_rl/ant_rough ${TAG}_expert; then
    python scripts/analysis/make_warmstart_checkpoint.py --src $BASELINE \
        --dst logs/rsl_rl/ant_rough/${TAG}_warmstart/model_0.pt --new_inputs 273 --std 0.2
    train --task Isaac-Ant-Rough-v0 --agent rsl_rl_warm_cfg_entry_point --run_name ${TAG}_expert \
        --resume --load_run ${TAG}_warmstart --checkpoint model_0.pt > $LOG/train_1_expert.log 2>&1
fi
EXPERT=$(last logs/rsl_rl/ant_rough ${TAG}_expert)

echo "== 2/4 teacher with privileged ground properties ($(date +%H:%M))"
if ! done_stage logs/rsl_rl/ant_rma ${TAG}_teacher; then
    python scripts/analysis/make_warmstart_checkpoint.py --src $EXPERT \
        --dst logs/rsl_rl/ant_rma/${TAG}_teacher_init/model_0.pt --new_inputs 323 --std 0.2
    train --task Isaac-Ant-RMA-v0 --run_name ${TAG}_teacher \
        --resume --load_run ${TAG}_teacher_init --checkpoint model_0.pt > $LOG/train_2_teacher.log 2>&1
fi
TEACHER=$(last logs/rsl_rl/ant_rma ${TAG}_teacher)

echo "== 3/4 student distillation ($(date +%H:%M))"
if ! done_stage logs/rsl_rl/ant_rma ${TAG}_student; then
    python scripts/analysis/make_student_checkpoints.py init --teacher $TEACHER \
        --dst logs/rsl_rl/ant_rma/${TAG}_student_init/model_0.pt
    train --task Isaac-Ant-RMA-v0 --agent rsl_rl_distillation_cfg_entry_point --run_name ${TAG}_student \
        --load_run ${TAG}_student_init --checkpoint model_0.pt > $LOG/train_3_student.log 2>&1
fi
STUDENT=$(last logs/rsl_rl/ant_rma ${TAG}_student)

echo "== 4/4 conservative PPO fine-tuning ($(date +%H:%M))"
if ! done_stage logs/rsl_rl/ant_rma ${TAG}_ft; then
    python scripts/analysis/make_student_checkpoints.py finetune_init --student $STUDENT --teacher $TEACHER \
        --dst logs/rsl_rl/ant_rma/${TAG}_ft_init/model_0.pt
    train --task Isaac-Ant-RMA-v0 --agent rsl_rl_finetune_safe_cfg_entry_point --run_name ${TAG}_ft \
        --resume --load_run ${TAG}_ft_init --checkpoint model_0.pt > $LOG/train_4_finetune.log 2>&1
fi
FINAL=$(last logs/rsl_rl/ant_rma ${TAG}_ft)

for c in "$EXPERT" "$TEACHER" "$STUDENT" "$FINAL"; do [ -f "$c" ] || { echo "missing checkpoint: '$c'"; exit 1; }; done
mkdir -p checkpoints
cp "$FINAL" checkpoints/ant_${TAG}.pt
printf "expert   %s\nteacher  %s\nstudent  %s\nfinal    %s -> checkpoints/ant_%s.pt\n" \
    "$EXPERT" "$TEACHER" "$STUDENT" "$FINAL" "$TAG" | tee $LOG/lineage.txt
echo "PhysX errors in training logs: $(cat $LOG/train_*.log | grep -c 'PhysX error' || true)"
echo "TRAINING DONE ($(date +%H:%M))"
