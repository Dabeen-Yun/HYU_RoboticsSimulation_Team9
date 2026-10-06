# 처음 보는 환경에서도 잘 걷는 Ant (로보틱스시뮬레이션 9조)

Isaac Lab 2.3.0 / Isaac Sim 5.1.0 기반 과제 저장소입니다.

- **목표:** Isaac-Ant-v0을 처음 보는 지형과 물성에서도 잘 걷게 만든다.
- **조건:** 보상 함수와 로봇 구조는 원본 그대로 
- **방법:** 교사-학생 학습
  - 교사는 정답 바닥 정보(마찰, 질량, 바닥 종류)를 보고 PPO로 학습.
  - 학생은 정답 없이 **발 접촉 센서 + 최근 8스텝 이력 + 높이맵**만으로 교사를 따라 배운다.
  - 마지막으로 학생을 원본 보상으로 조심스럽게 미세조정.
  - 배포되는 것은 학생 정책 (`checkpoints/ant_final.pt`).

## 결과 (공식 평가: `play_one_episode.py --seed 24 --num_envs 100`, 원본 Ant 보상)

| 환경 | Baseline (`checkpoints/ant_baseline.pt`) | **최종 정책 (`checkpoints/ant_final.pt`)** | 차이 |
|---|---|---|---|
| 기존 환경 (Isaac-Ant-v0) | 129.57 ± 29.68 | **126.72 ± 28.42** | −2.9 (p = 0.49, 차이 없음) |
| 처음 보는 환경 (우리 검증 환경) | 36.28 ± 18.29 | **69.29 ± 21.90** | **+33.0 (+91%)**, p = 8×10⁻²⁴ |
| 새 환경 예시 (`ant_new_env_cfg.py` 기본값) | 12.30 ± 12.06 | **38.20 ± 31.90** | **+25.9 (약 3배)**, p = 6×10⁻¹² |

---

## 1. 설치

필요 환경: Ubuntu 22.04, NVIDIA GPU (VRAM 8 GB 이상), NVIDIA 드라이버 535 이상, conda.
개발에 쓴 환경은 RTX 3070 Ti 8 GB, 드라이버 580입니다.

```bash
# 1) Python 3.11 환경
conda create -n isaaclab python=3.11 -y
conda activate isaaclab

# 2) PyTorch 2.7 (CUDA 12.8)와 Isaac Sim 5.1
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# 3) 이 저장소와 Isaac Lab 확장 설치 (rsl-rl-lib 3.0.1 포함)
git clone https://github.com/Dabeen-Yun/HYU_RoboticsSimulation_Team9.git
cd HYU_RoboticsSimulation_Team9
./isaaclab.sh -i rsl_rl

# 4) 설치 확인: Isaac-Ant-... 목록이 나오면 성공
./isaaclab.sh -p scripts/environments/list_envs.py | grep Isaac-Ant
```

## 2. 결과 재현

```bash
./scripts/final/eval_final.sh
```

Baseline과 최종 정책을 기존 환경과 처음 보는 환경에서 각각 평가해 위 결과표의 숫자를 출력합니다.
- 전체 출력은 `results/final/official_*.txt`에 저장.
과제 안내의 공식 평가 형식(`--seed 24 --num_envs 100`, `--task`와 `--checkpoint`는 정책에 맞게)으로 직접 실행하려면 아래와 같습니다. 우리 정책에는 `--agent rsl_rl_finetune_cfg_entry_point`를 추가합니다.

```bash
# 기존 환경 (Isaac-Ant-v0)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py --headless --seed 24 --num_envs 100 \
    --task Isaac-Ant-v0 --checkpoint checkpoints/ant_baseline.pt
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py --headless --seed 24 --num_envs 100 \
    --task Isaac-Ant-RMA-Flat-v0 --agent rsl_rl_finetune_cfg_entry_point --checkpoint checkpoints/ant_final.pt

# 처음 보는 환경 (학습에 쓰지 않은 지형과 물성)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py --headless --seed 24 --num_envs 100 \
    --task Isaac-Ant-Test-Blind-v0 --checkpoint checkpoints/ant_baseline.pt
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py --headless --seed 24 --num_envs 100 \
    --task Isaac-Ant-RMA-Test-v0 --agent rsl_rl_finetune_cfg_entry_point --checkpoint checkpoints/ant_final.pt
```

**처음 보는 환경의 구성** (`ant_rough_env_cfg.py`의 `ANT_TEST_TERRAINS_CFG`, `AntTestEventCfg`)
- **지형:** 파도와 장애물(학습에 없는 종류), 거친 요철 최대 10 cm(학습 범위의 1.7배)
  - 바닥은 삼각형 메쉬만 쓴다.
  - 지형 생성 seed는 24로 고정했다.
- **물성:** 마찰 0.3~0.7, 질량 ×0.7~1.3 (학습 범위보다 넓음), 밀기 없음

화면으로 보려면 `--headless`를 빼면 됩니다. 영상을 저장하려면 `--video --video_length 960 --device cpu`를 붙입니다(GPU에서는 높이맵 센서 때문에 녹화가 실패합니다).

## 3. 새로운 환경에서 실행

1. `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_new_env_cfg.py`를 엽니다.
2. `EDIT HERE` 구역만 바꿉니다.

   | 변수 | 의미 | 예시 |
   |---|---|---|
   | `NEW_TERRAIN` | 지형 생성기 (`None`이면 원래 평면 바닥) | 계단, 경사, 파도, 장애물, 요철 등 `isaaclab.terrains.Hf*TerrainCfg` |
   | `GROUND_FRICTION` | 바닥 마찰 (`None`이면 1.0) | `0.2` (미끄러운 바닥) |
   | `TORSO_MASS_SCALE` | 몸통 질량 배율 (`None`이면 원래 값) | `1.5` |
   | `PUSH` | 주기적으로 밀기 (`None`이면 없음) | `((2.0, 4.0), 1.0)` → 2~4초마다 최대 1 m/s |

   기본값은 예시 환경입니다. 학습에 없던 역피라미드 계단·경사와 요철에, 마찰 0.5와 몸통 질량 ×1.3을 적용했습니다.
3. 평가합니다.

   ```bash
   ./scripts/final/eval_final.sh new
   ```

   같은 환경이 Baseline용(`Isaac-Ant-New-v0`)과 우리 정책용(`Isaac-Ant-RMA-New-v0`)으로 등록되어 있으므로, 개별 명령으로 실행해도 됩니다.

   ```bash
   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py --headless --seed 24 --num_envs 100 \
       --task Isaac-Ant-New-v0 --checkpoint checkpoints/ant_baseline.pt
   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py --headless --seed 24 --num_envs 100 \
       --task Isaac-Ant-RMA-New-v0 --agent rsl_rl_finetune_cfg_entry_point --checkpoint checkpoints/ant_final.pt
   ```

**이미 만들어 둔 다른 환경 클래스를 쓰려면**
- Isaac-Ant-v0 기반 환경 설정 클래스가 있으면, 아래 한 줄로 우리 센서를 붙인 버전을 만들 수 있습니다. 만든 클래스를 `__init__.py`에 `Isaac-Ant-RMA-New-v0`와 같은 방식으로 등록한 뒤, `--agent rsl_rl_finetune_cfg_entry_point`로 평가합니다.

  ```python
  from .ant_rma_bundle_eval_cfg import with_rma_sensors
  MyRMAEnvCfg = with_rma_sensors(MyEnvCfg, "MyRMAEnvCfg")
  ```
- 이때 지형 변경은 `__post_init__` 안에서 해야 합니다. 래퍼가 장면(scene)을 교체하기 때문입니다.
- 보상과 종료 조건은 바꾸지 마세요. 그래야 점수가 원본 기준으로 비교됩니다.

**바로 쓸 수 있는 다른 평가 환경** (팀원이 만든 환경, `ant_robust_env_cfg.py`). 우리 정책용 task에도 `--agent rsl_rl_finetune_cfg_entry_point`를 붙입니다.

| Baseline용 task | 우리 정책용 task | 조건 |
|---|---|---|
| `Isaac-Ant-Test-LowFriction-v0` | `Isaac-Ant-RMA-Bundle-LowFriction-v0` | 마찰 0.25 |
| `Isaac-Ant-Test-Heavy-v0` | `Isaac-Ant-RMA-Bundle-Heavy-v0` | 몸통 ×1.8 |
| `Isaac-Ant-Test-WeakMotor-v0` | `Isaac-Ant-RMA-Bundle-WeakMotor-v0` | 모터 토크 ×2/3 |
| `Isaac-Ant-Test-Rough-v0` | `Isaac-Ant-RMA-Bundle-Rough-v0` | 다른 배치의 요철 지형 (≤8 cm) |
| `Isaac-Ant-Test-Push-v0` | `Isaac-Ant-RMA-Bundle-Push-v0` | 2~4초마다 ±1.5 m/s 밀기 |
| `Isaac-Ant-Test-Heading-v0` | `Isaac-Ant-RMA-Bundle-Heading-v0` | 출발 방향 ±180° |
| `Isaac-Ant-Test-Combined-v0` | `Isaac-Ant-RMA-Bundle-Combined-v0` | 위 조건 여러 개 동시 적용 |

## 4. 왜 정책마다 task가 다른가

- 우리 정책은 원본 관측(60차원)에 센서 정보를 더 받습니다: 높이맵 273, 발 접촉 16, 최근 8스텝 이력 304 → **653차원**.
- 그래서 같은 환경이라도 센서가 붙은 버전(`Isaac-Ant-RMA-*`)으로 실행하고, `--agent rsl_rl_finetune_cfg_entry_point`로 최종 정책의 설정을 불러와야 합니다.
- 두 버전은 **지형, 물성, 보상, 종료 조건이 모두 같고** 센서만 다릅니다.

| 환경 | Baseline (60차원) | 우리 정책 (653차원) |
|---|---|---|
| 기존 환경 | `Isaac-Ant-v0` | `Isaac-Ant-RMA-Flat-v0` |
| 처음 보는 환경 | `Isaac-Ant-Test-Blind-v0` | `Isaac-Ant-RMA-Test-v0` |
| 새 환경 (3절) | `Isaac-Ant-New-v0` | `Isaac-Ant-RMA-New-v0` |

## 5. 처음부터 다시 학습 (약 1시간, RTX 3070 Ti 기준)

```bash
./scripts/final/train_final.sh
```

강의 Baseline(`checkpoints/ant_baseline.pt`, 4096 envs × 1000 iter, seed 42)에서 출발해 네 단계를 차례로 학습합니다.

| 단계 | 내용 | 학습 환경 |
|---|---|---|
| 1 | 높이맵을 붙인 지형 정책 (Baseline에서 이어 학습) | `Isaac-Ant-Rough-v0` |
| 2 | 교사: 정답 바닥 정보까지 보고 PPO | `Isaac-Ant-RMA-v0` |
| 3 | 학생: 정답 없이 교사 행동을 따라 배움 (DAgger) | `Isaac-Ant-RMA-v0` |
| 4 | 학생을 원본 보상으로 미세조정 (학습률 5e-5) | `Isaac-Ant-RMA-v0` |

- 단계마다 **4096 envs × 1000 iter, 학습 seed 42**입니다.
- 결과는 `checkpoints/ant_final.pt`로 저장됩니다. 4단계의 마지막 iteration을 그대로 씁니다.
- 학습 로그는 `results/final/train_*.log`, 각 단계의 체크포인트 경로는 `results/final/lineage.txt`에 남습니다.
- 학습 후 2절의 `eval_final.sh`로 평가합니다.
- GPU 연산은 완전히 결정적이지 않아서, 다시 학습하면 점수가 몇 점 달라질 수 있습니다. 같은 숫자를 확인하려면 저장소에 포함된 `checkpoints/ant_final.pt`로 평가하면 됩니다.
- GPU 메모리가 8 GB보다 작으면 `NUM_ENVS=2048 ./scripts/final/train_final.sh`처럼 줄일 수 있습니다. 다만 그러면 학습 조건이 위와 달라집니다.

## 6. 파일 구조

```
checkpoints/
  ant_final.pt          최종 정책 (5절 파이프라인 결과)
  ant_baseline.pt       강의 Baseline
scripts/final/
  train_final.sh        최종 정책 학습 (4단계)
  eval_final.sh         공식 평가 (우리 환경 / new: 새 환경)
source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/
  ant_rough_env_cfg.py  학습 지형, 처음 보는 환경, 높이맵 센서
  ant_rma_env_cfg.py    발 접촉 센서, 이력, 정답 물성(교사), Isaac-Ant-RMA-* 환경
  ant_new_env_cfg.py    새 평가 환경 템플릿 (3절)
  ant_robust_env_cfg.py 팀원의 평가 환경 모음
  agents/rsl_rl_ppo_cfg.py  교사, 학생, 미세조정 학습 설정
  __init__.py           task ID 등록
scripts/analysis/       분석·그림 스크립트 (발표 자료용)
results/                실험 기록 (results/README.md), 그림, 평가 원자료
```

전체 실험 과정과 비교 실험(높이맵 단독, 전문가 전환, 넓은 범위 학습, IMU 등)은 [`results/README.md`](results/README.md)에 정리되어 있습니다.
