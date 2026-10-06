#!/usr/bin/env bash
# Evaluate the lecture baseline and our final policy with the official protocol
# (play_one_episode.py, --seed 24 --num_envs 100, original Isaac-Ant-v0 reward).
#
#   ./scripts/final/eval_final.sh            original env + our held-out env (the numbers in the README)
#   ./scripts/final/eval_final.sh new        the environment defined in ant_new_env_cfg.py
#
# Each line of output is "<policy> <env>: [RESULT] Episode reward total: mean=..., std=...".
# Full outputs are saved to results/final/official_<policy>_<env>.txt.
set -u
cd "$(dirname "$0")/../.."
BASELINE=${BASELINE:-checkpoints/ant_baseline.pt}
FINAL=${FINAL:-checkpoints/ant_final.pt}
AGENT="--agent rsl_rl_finetune_cfg_entry_point"   # our policy: 653-D observations (sensors + history)
OUT=results/final
mkdir -p $OUT

play() {  # name env task checkpoint [extra args]
    echo -n "$1 $2: "
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py --headless --seed 24 --num_envs 100 \
        --task "$3" --checkpoint "$4" "${@:5}" 2>&1 | grep -E "\[RESULT\]|Completed first" | tee $OUT/official_$1_$2.txt \
        | grep "reward total"
}

if [ "${1:-ours}" = "new" ]; then
    play baseline new Isaac-Ant-New-v0     $BASELINE
    play final    new Isaac-Ant-RMA-New-v0 $FINAL $AGENT
else
    # baseline: 60-D observation tasks; final: the same envs + our sensors (Isaac-Ant-RMA-*)
    play baseline original Isaac-Ant-v0            $BASELINE
    play final    original Isaac-Ant-RMA-Flat-v0   $FINAL $AGENT
    play baseline unseen   Isaac-Ant-Test-Blind-v0 $BASELINE
    play final    unseen   Isaac-Ant-RMA-Test-v0   $FINAL $AGENT
fi
