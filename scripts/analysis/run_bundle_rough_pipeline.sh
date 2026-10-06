#!/usr/bin/env bash
# [ant_robust_bundle] Our method trained from scratch on the bundle's Rough pair, compared with the lecture baseline.
#   train : Isaac-Ant-RMA-Bundle-DR-Rough-v0  (= Isaac-Ant-DR-Rough-v0 + our sensors)
#   test  : Isaac-Ant-RMA-Bundle-Rough-v0     (= Isaac-Ant-Test-Rough-v0 + our sensors)
#   every stage: 4096 envs x 1000 iter, seed 42;  evaluation: seed 24, 100 envs (play_one_episode protocol)
set -u
cd "$(dirname "$0")/../.."
D=logs/rsl_rl/ant_rma
OUT=results/bundle_rough
TRAIN=Isaac-Ant-RMA-Bundle-DR-Rough-v0
mkdir -p $OUT
train() { ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task $TRAIN --headless --num_envs 4096 --seed 42 \
              --max_iterations 1000 "$@"; }
ev() { timeout -s KILL 300 ./isaaclab.sh -p scripts/analysis/evaluate_ant.py --seed 24 --num_envs 100 --headless "$@" 2>&1 \
           | grep "\[RESULT\]"; }

echo "== 1/3 teacher PPO from scratch ($(date +%H:%M))"
train --run_name bundle_rough_teacher > $OUT/train_teacher.log 2>&1
TEACHER=$(ls -d $D/*_bundle_rough_teacher | tail -1)/model_999.pt

echo "== 2/3 student distillation ($(date +%H:%M))"
python scripts/analysis/make_student_checkpoints.py init --teacher $TEACHER --dst $D/bundle_rough_student_init/model_0.pt
train --agent rsl_rl_distillation_cfg_entry_point --run_name bundle_rough_student --load_run bundle_rough_student_init \
    --checkpoint model_0.pt > $OUT/train_student.log 2>&1
STUDENT=$(ls -d $D/*_bundle_rough_student | tail -1)/model_999.pt

echo "== 3/3 conservative PPO fine-tuning ($(date +%H:%M))"
python scripts/analysis/make_student_checkpoints.py finetune_init --student $STUDENT --teacher $TEACHER \
    --dst $D/bundle_rough_ft_init/model_0.pt
train --agent rsl_rl_finetune_safe_cfg_entry_point --run_name bundle_rough_ft --resume --load_run bundle_rough_ft_init \
    --checkpoint model_0.pt > $OUT/train_ft.log 2>&1
FINAL=$(ls -d $D/*_bundle_rough_ft | tail -1)/model_999.pt
python scripts/analysis/make_student_checkpoints.py export --student $STUDENT \
    --flat checkpoints/ant_baseline.pt --dst_dir $D/bundle_rough_student_export > /dev/null
cp $FINAL checkpoints/ant_bundle_rough_final.pt

grep -c "PhysX error" $OUT/train_*.log

echo "== evaluation ($(date +%H:%M))"
echo -n "baseline  test: "; ev --task Isaac-Ant-Test-Rough-v0 --checkpoint checkpoints/ant_baseline.pt --out $OUT/baseline_test.npz
echo -n "teacher   test: "; ev --task Isaac-Ant-RMA-Bundle-Rough-v0 --checkpoint $TEACHER --out $OUT/teacher_test.npz
echo -n "student   test: "; ev --task Isaac-Ant-RMA-Bundle-Rough-v0 --agent rsl_rl_student_cfg_entry_point \
    --checkpoint $D/bundle_rough_student_export/student_plain/model_0.pt --out $OUT/student_test.npz
echo -n "ours      test: "; ev --task Isaac-Ant-RMA-Bundle-Rough-v0 --agent rsl_rl_finetune_cfg_entry_point \
    --checkpoint checkpoints/ant_bundle_rough_final.pt --out $OUT/ours_test.npz
echo -n "baseline  original: "; ev --task Isaac-Ant-v0 --checkpoint checkpoints/ant_baseline.pt --out $OUT/baseline_original.npz
echo -n "ours      original: "; ev --task Isaac-Ant-RMA-Flat-v0 --agent rsl_rl_finetune_cfg_entry_point \
    --checkpoint checkpoints/ant_bundle_rough_final.pt --out $OUT/ours_original.npz

echo "== official play_one_episode.py output"
for line in "baseline|Isaac-Ant-Test-Rough-v0|rsl_rl_cfg_entry_point|checkpoints/ant_baseline.pt" \
            "ours|Isaac-Ant-RMA-Bundle-Rough-v0|rsl_rl_finetune_cfg_entry_point|checkpoints/ant_bundle_rough_final.pt"; do
    IFS='|' read -r name task agent ckpt <<< "$line"
    echo "-- $name"
    timeout -s KILL 300 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py --task $task --agent $agent \
        --seed 24 --num_envs 100 --headless --checkpoint $ckpt 2>&1 | grep -E "\[RESULT\]|Completed first" | tee $OUT/official_$name.txt
done
echo "PIPELINE DONE ($(date +%H:%M))"
