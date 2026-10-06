# 과제 가이드: 처음 보는 환경에서도 잘 걷는 Ant 만들기

## 핵심 요약 (체크리스트)

- [ ] 하드웨어 구조(링크 길이, 관절 개수 등)는 건드리지 않는다. Python 코드 / cfg / YAML만 수정한다.
- [ ] Ant USD 파일은 수정하지 않는다. (예외: 센서 추가를 위한 수정은 허용)
- [ ] 연구 문제 정의 → 가설 수립 → 검증 실험 설계 → 결과 분석의 흐름을 갖춘다.
- [ ] 학습에 쓰지 않은 **새 평가 환경**을 팀에서 직접 만든다.
- [ ] `play_one_episode.py`로 평가한다. 인자는 `--seed 24 --num_envs 100`이다.
- [ ] 보상 함수를 바꿨다면, 평가 출력은 **원본 Isaac-Ant-v0 보상 기준**이 되도록 `play_one_episode.py`를 수정한다.
- [ ] 출력된 reward mean/std를 5분 발표 PPT에 기입한다.

---

## 1. 로봇 변형 범위

- 로봇의 하드웨어 구조(링크 길이, 관절 개수 등)는 변경할 수 없습니다.
- 로봇의 소프트웨어적 제어 요소(PD gain, 토크 한계 등)만 변경할 수 있습니다.
- 즉, Python 코드나 파라미터 설정 파일(YAML, cfg)만 수정해 주세요.
- Isaac-Ant-v0의 USD 파일은 수정하지 않는 것을 원칙으로 하되, 센서 추가를 위한 수정은 허용합니다.
- 로봇의 센서, 물성치(P gain, D gain 등), 학습 모델 아키텍처, 보상 함수 등은 자유롭게 설계하셔도 됩니다.

## 2. 평가 기준

### (1) 실험 설계 및 결과 분석

- 조교진이 가장 중요하게 보는 것은 **"연구적인 실험 설계와 결과 분석"**입니다.
- 과제 목표는 **"처음 보는 환경에서도 잘 걷는 Ant 만들기"**입니다. 다음 흐름을 논리적으로 탄탄하게 구성할수록 높은 점수를 받습니다.
  - **연구 문제 정의** (예: 처음 보는 환경에서 Ant의 보행 성능은 왜 저하되는가?)
  - 문제 해결을 위한 **가설 수립**
  - 가설을 검증하기 위한 **실험 설계와 결과 분석**
- reward 순위만으로 평가하지 않습니다. 실험 결과는 가설을 입증하기 위한 수단이며, 실험 설계의 타당성과 결과 분석의 논리적 흐름을 종합하여 평가합니다.
- 본 과목은 융합 학문 과목이라 과제의 자유도를 최대한 열어 두었습니다. 연구 문제를 잘 정의하고, 이를 해결하는 과정을 보여 주면 됩니다.
- 세부 채점 기준(감점 사유 등)은 과제 점수와 함께 공개됩니다.

### (2) 발표

- 보고서와 발표의 완성도(내용 전달력, 시간 준수 등)를 평가합니다. (5분 발표)

### (3) 독창성

- 유승환, 이강권 조교가 팀별 독창성을 상/중/하로 평가합니다.

## 3. 팀별 자체 평가 방법 (발표 PPT 기입용)

학습에 사용하지 않은 새로운 환경을 팀에서 자체적으로 만든 후, 그 환경에서
`~/IsaacLab_RS/scripts/reinforcement_learning/rsl_rl/play_one_episode.py`를 실행하여 평가합니다.

- CLI 인자는 **`--seed 24`**, **`--num_envs 100`**으로 고정합니다.
- `--task`, `--checkpoint`는 팀의 환경과 학습 결과에 맞게 바꿉니다.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
    --task Isaac-Ant-v0 \
    --seed 24 \
    --num_envs 100 \
    --checkpoint logs/rsl_rl/ant/<log 폴더 이름>/model_999.pt
```

실행이 끝나면 100개 환경의 "에피소드 누적 보상" 평균(mean)과 표준편차(std)가 출력됩니다. 이 값을 PPT에 기입합니다.

```
[INFO] All 100 environments finished their first episode.
[INFO] Completed first episodes: 100/100
[RESULT] Episode reward total: mean=123.059077, std=27.972435
[RESULT] Episode steps: mean=913.730000, std=181.523709
```

> ⚠️ "Isaac-Ant-v0" 환경의 보상 함수를 변경한 팀은, `play_one_episode.py` 실행 시
> **원본 Isaac-Ant-v0 환경의 보상 함수 기준**으로 평균과 표준편차가 출력되도록 수정해야 합니다.

### 원본 Isaac-Ant-v0 보상 함수 (참고용)

출처: `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_env_cfg.py`

| 항목 | 함수 | weight | 파라미터 |
|---|---|---|---|
| progress | `progress_reward` | 1.0 | target_pos=(1000, 0, 0) |
| alive | `is_alive` | 0.5 | |
| upright | `upright_posture_bonus` | 0.1 | threshold=0.93 |
| move_to_target | `move_to_target_bonus` | 0.5 | threshold=0.8, target_pos=(1000, 0, 0) |
| action_l2 | `action_l2` | -0.005 | |
| energy | `power_consumption` | -0.05 | gear_ratio=15 |
| joint_pos_limits | `joint_pos_limits_penalty_ratio` | -0.1 | threshold=0.99, gear_ratio=15 |

종료 조건: time_out (16초), 몸통 높이 < 0.31

---

## 원본 코드 백업 (2026-10-04)

- **git 태그 `original-baseline`**: 수정 전 전체 코드 (커밋 e83a5d2)
  - 파일 하나 복원: `git checkout original-baseline -- <파일경로>`
  - 원본과 비교: `git diff original-baseline -- <파일경로>`
- **git 태그 `result-rma-student`** (2026-10-05): 교사-학생 최종 결과 스냅샷 (`result-gated-v4` 위에 쌓음, main과 별개)
  - 추가로 포함: `ant_rma_env_cfg.py`, 교사 체크포인트, 학생 체크포인트(단독 `student_plain`, 전환 `student_gated`), 마찰 스윕 결과
- **git 태그 `result-gated-v4`** (2026-10-05): 최종 결과 스냅샷 (main 브랜치와는 별개로 저장)
  - 포함: Ant 설정 코드, `scripts/analysis/`, `results/` 전체, 이 가이드, 체크포인트 3개 (gated_v4, v4_warm `model_1999.pt`, baseline `model_999.pt`)
  - 내용 보기: `git show --stat result-gated-v4`
  - 파일 하나 복원: `git checkout result-gated-v4 -- <파일경로>`
  - 전체를 새 폴더로 꺼내기: `git worktree add ../IsaacLab_RS_result result-gated-v4`
- **`backup_original/` 폴더**: Ant 관련 핵심 파일 복사본 (원래와 같은 폴더 구조)
  - 환경/보상: `manager_based/classic/ant/`, `classic/humanoid/mdp/`
  - direct 버전: `direct/ant/`, `direct/locomotion/`
  - 로봇: `isaaclab_assets/robots/ant.py`
  - 스크립트: `train.py`, `play.py`, `play_one_episode.py`, `cli_args.py`

---

## 학습/평가 task (2026-10-05 갱신)

설정 파일: `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_rough_env_cfg.py`
**보상은 원본 `AntEnvCfg` 그대로이며 절대 수정하지 않습니다.**
평가 환경은 종료 조건도 원본 그대로 씁니다. 학습 환경만 몸통 높이를 지형 기준으로 잽니다.
**전체 결과와 그림:** `results/README.md`

| task | 용도 | 관측 | 지형/물성 |
|---|---|---|---|
| `Isaac-Ant-v0` | 원본 (평면 바닥) | 60 | 원래 ground plane, 마찰 1.0 |
| `Isaac-Ant-Rough-v0` | **학습용** | 333 (60 + 높이맵 273) | 평면 바닥 30% + 요철, 경사, 계단, 박스 / 마찰 0.4~1.2, 질량 ×0.8~1.2, 밀기, 초기 yaw ±30° |
| `Isaac-Ant-Rough-Rehearsal-v0` | 학습용 변형 (v5 실험) | 333 | 위와 같음. 단 평면 칸은 랜덤화 없이 원래 조건 |
| `Isaac-Ant-Rough-Flat-v0` | **기존 환경 평가** (높이맵 정책) | 333 | Isaac-Ant-v0과 동일 + 높이 센서 |
| `Isaac-Ant-Rough-Test-v0` | **처음 보는 환경 평가** (높이맵 정책) | 333 | 파도, 장애물, 더 거친 요철 / 마찰 0.3~0.7, 질량 ×0.7~1.3, 지형 seed 24 |
| `Isaac-Ant-Test-Blind-v0` | 처음 보는 환경 평가 (높이맵 없는 정책) | 60 | 위와 같은 평가 환경 |
| `Isaac-Ant-Rough-Blind-v0` | 비교군 학습 (높이맵 없음) | 60 | Rough-v0과 동일 |
| `Isaac-Ant-Blind-Flat-v0` | 대조 실험 | 60 | 평평한 삼각형 메쉬 바닥 (접촉 모델 차이 확인용) |

### 최종 정책: `logs/rsl_rl/ant_rma/final/model_0.pt` (`--agent rsl_rl_finetune_cfg_entry_point`)
교사-학생 학습(높이맵, 발 접촉 센서, 8스텝 이력)을 한 뒤, 조심스러운 PPO 미세조정을 더한 단일 정책입니다. 자세한 내용은 `results/README.md` 13절에 있습니다. IMU 실험은 이득이 없어 코드에서 제거했습니다(12절).
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py --task Isaac-Ant-RMA-Test-v0 \
    --agent rsl_rl_finetune_cfg_entry_point --seed 24 --num_envs 100 --checkpoint logs/rsl_rl/ant_rma/final/model_0.pt
```

| 정책 | 기존 환경 | 처음 보는 환경 |
|---|---|---|
| Baseline `logs/rsl_rl/ant/2026-09-28_14-21-51_ant_baseline/model_999.pt` | 129.57 ± 29.68 | 36.28 ± 18.29 |
| 1차: 높이맵 + 전문가 전환 (`ant_rough/gated_v4`) | 130.93 | 60.53 |
| 2차: 교사-학생 (`ant_rma/student_plain`, 시드 42) | 138.35 | 64.64 |
| **최종: 교사-학생 + 미세조정 (`ant_rma/final`)** | **147.14 ± 29.21** | **59.80 ± 29.88** |
| 최종 방법의 3회 평균 (시드 42 / 1 / 2) | 138.4 ± 9.1 | 64.0 ± 3.7 |

| RMA task | 용도 |
|---|---|
| `Isaac-Ant-RMA-v0` | 교사와 학생 학습 (Rough-v0 + 발 접촉 센서 + 이력 + 정답 물성, 마찰은 개체마다 하나) |
| `Isaac-Ant-RMA-Flat-v0` / `Isaac-Ant-RMA-Test-v0` | 기존 / 처음 보는 환경 평가 (Rough-Flat / Rough-Test와 같은 환경) |

### 학습 시 주의
- GPU가 8 GB라서 `--num_envs 2048`로 학습합니다. 4096개로 돌리면 PhysX가 메모리 부족으로 멈춥니다.
- 높이 센서(RayCaster)를 쓰는 환경에서 `--video`로 녹화하려면 `--device cpu`가 필요합니다. GPU 물리에서는 CUDA 오류가 납니다.
