#!/usr/bin/env bash
# Two ways to reduce the teacher -> student transfer loss, tried on the seed-42 student:
#   E1 conservative PPO fine-tuning (lr 5e-5 fixed, critic from the teacher)
#   E2 continued distillation (+1500 iter, lr 3e-4)
set -u
cd "$(dirname "$0")/../.."
D=logs/rsl_rl/ant_rma
BASE=logs/rsl_rl/ant/2026-09-28_14-21-51_ant_baseline/model_999.pt
ev() { timeout -s KILL 300 ./isaaclab.sh -p scripts/analysis/evaluate_ant.py --seed 24 --num_envs 100 --headless "$@" 2>&1 | grep "\[RESULT\]"; }

echo "===== E1 safe fine-tune ($(date +%H:%M))"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Ant-RMA-v0 --agent rsl_rl_finetune_safe_cfg_entry_point \
    --headless --num_envs 2048 --seed 42 --max_iterations 1500 --run_name ftsafe_s42 --resume --load_run ft_s42_init \
    --checkpoint model_0.pt > results/finetune/train_ftsafe_s42.log 2>&1
C=$(ls -d $D/*_ftsafe_s42 | tail -1)
for it in 500 1499; do for env in Flat Test; do e=$(echo $env | tr A-Z a-z)
    echo -n "E1 iter $it $e: "; ev --task Isaac-Ant-RMA-$env-v0 --agent rsl_rl_finetune_cfg_entry_point --checkpoint $C/model_$it.pt \
        --out results/finetune/ftsafe_s42_it${it}_$e.npz; done; done

echo "===== E2 longer distillation ($(date +%H:%M))"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Ant-RMA-v0 --agent rsl_rl_distillation_long_cfg_entry_point \
    --headless --num_envs 2048 --seed 42 --max_iterations 1500 --run_name distlong_s42 --resume \
    --load_run 2026-10-05_10-37-15_student --checkpoint model_1499.pt > results/finetune/train_distlong_s42.log 2>&1
S=$(ls -d $D/*_distlong_s42 | tail -1)
LAST=$(ls $S | grep model_ | sed 's/model_//;s/.pt//' | sort -n | tail -1)
python scripts/analysis/make_student_checkpoints.py export --student $S/model_$LAST.pt --flat $BASE --dst_dir $D/distlong_s42 > /dev/null
for env in Flat Test; do e=$(echo $env | tr A-Z a-z)
    echo -n "E2 $e: "; ev --task Isaac-Ant-RMA-$env-v0 --agent rsl_rl_student_cfg_entry_point \
        --checkpoint $D/distlong_s42/student_plain/model_0.pt --out results/finetune/distlong_s42_$e.npz; done
grep "behavior loss" results/finetune/train_distlong_s42.log | sed -n '1p;$p'
echo "VARIANTS DONE ($(date +%H:%M))"
