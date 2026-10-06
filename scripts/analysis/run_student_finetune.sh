#!/usr/bin/env bash
# Student PPO fine-tuning (asymmetric actor-critic) for the three teacher-student runs (seeds 42, 1, 2).
#   actor  <- distilled student (deployable observations)
#   critic <- teacher critic (privileged observations, training only)
# Results: results/finetune/ft_s<seed>_<flat|test>.npz
set -u
cd "$(dirname "$0")/../.."
D=logs/rsl_rl/ant_rma
ITERS=${ITERS:-1500}
mkdir -p results/finetune

declare -A TEACHER STUDENT
TEACHER[42]=$D/2026-10-05_10-14-09_teacher/model_1999.pt;     STUDENT[42]=$D/2026-10-05_10-37-15_student/model_1499.pt
TEACHER[1]=$D/2026-10-05_15-21-07_rma_s1_teacher/model_1999.pt; STUDENT[1]=$D/2026-10-05_15-42-35_rma_s1_student/model_1499.pt
TEACHER[2]=$D/2026-10-05_17-00-53_rma_s2_teacher/model_1999.pt; STUDENT[2]=$D/2026-10-05_17-22-26_rma_s2_student/model_1499.pt

for seed in ${SEEDS:-42 1 2}; do
    echo "===== ft_s$seed ($(date +%H:%M))"
    python scripts/analysis/make_student_checkpoints.py finetune_init --student "${STUDENT[$seed]}" \
        --teacher "${TEACHER[$seed]}" --dst $D/ft_s${seed}_init/model_0.pt > /dev/null
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Ant-RMA-v0 \
        --agent rsl_rl_finetune_cfg_entry_point --headless --num_envs 2048 --seed $seed --max_iterations $ITERS \
        --run_name ft_s${seed} --resume --load_run ft_s${seed}_init --checkpoint model_0.pt \
        > results/finetune/train_ft_s${seed}.log 2>&1
    CKPT=$(ls -d $D/*_ft_s${seed} | tail -1)/model_$((ITERS - 1)).pt
    for env in Flat Test; do
        e=$(echo $env | tr A-Z a-z)
        echo -n "ft_s$seed $e: "
        timeout -s KILL 300 ./isaaclab.sh -p scripts/analysis/evaluate_ant.py --task Isaac-Ant-RMA-$env-v0 \
            --agent rsl_rl_finetune_cfg_entry_point --seed 24 --num_envs 100 --headless --checkpoint "$CKPT" \
            --out results/finetune/ft_s${seed}_$e.npz 2>&1 | grep "\[RESULT\]"
    done
done
echo "FINETUNE DONE ($(date +%H:%M))"
