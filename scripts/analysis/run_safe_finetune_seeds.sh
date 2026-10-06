#!/usr/bin/env bash
# Conservative PPO fine-tuning (lr 5e-5 fixed) of the seed-1 and seed-2 students, evaluated at 500 and 1500 iterations.
set -u
cd "$(dirname "$0")/../.."
D=logs/rsl_rl/ant_rma
ev() { timeout -s KILL 300 ./isaaclab.sh -p scripts/analysis/evaluate_ant.py --seed 24 --num_envs 100 --headless "$@" 2>&1 | grep "\[RESULT\]"; }
for seed in ${SEEDS:-1 2}; do
    echo "===== ftsafe_s$seed ($(date +%H:%M))"
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Ant-RMA-v0 --agent rsl_rl_finetune_safe_cfg_entry_point \
        --headless --num_envs 2048 --seed $seed --max_iterations 1500 --run_name ftsafe_s$seed --resume --load_run ft_s${seed}_init \
        --checkpoint model_0.pt > results/finetune/train_ftsafe_s$seed.log 2>&1
    C=$(ls -d $D/*_ftsafe_s$seed | tail -1)
    for it in 500 1499; do for env in Flat Test; do e=$(echo $env | tr A-Z a-z)
        echo -n "ftsafe_s$seed iter $it $e: "; ev --task Isaac-Ant-RMA-$env-v0 --agent rsl_rl_finetune_cfg_entry_point \
            --checkpoint $C/model_$it.pt --out results/finetune/ftsafe_s${seed}_it${it}_$e.npz; done; done
done
echo "SAFE SEEDS DONE ($(date +%H:%M))"
