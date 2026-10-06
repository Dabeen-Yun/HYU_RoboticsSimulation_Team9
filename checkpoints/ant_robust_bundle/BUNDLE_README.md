# Ant 강건성 실험: 환경 파일과 체크포인트

IsaacLab 2.3.0 (IsaacLab_RS)과 Isaac Sim 5.1.0 기준입니다. 추가로 설치할 패키지는 없습니다.

## 1. 설치

`env/`에 있는 파일 두 개를 아래 폴더에 복사합니다.

```
IsaacLab_RS/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/
├── ant_robust_env_cfg.py   ← 새 파일 (모든 환경 정의)
└── __init__.py             ← 덮어쓰기 (Task ID 등록 추가)
```

- `__init__.py`를 이미 수정해서 쓰고 있다면, 덮어쓰지 말고 `env/__init__.py.diff`에 있는 추가분만 붙여 넣으세요.
- 원본 `Isaac-Ant-v0`는 바뀌지 않습니다.
- `isaaclab.sh -i` 같은 재설치는 필요 없습니다. editable 설치라서 파일만 복사하면 바로 반영됩니다.

설치를 확인하려면 아래 명령을 실행합니다. `Isaac-Ant-DR-...`, `Isaac-Ant-Test-...` 같은 ID가 보이면 된 것입니다.

```bash
./isaaclab.sh -p scripts/environments/list_envs.py | grep Isaac-Ant
```

## 2. Task ID

관측(60차원)과 행동(8차원)은 모두 원본 Isaac-Ant-v0와 같습니다. 그래서 어떤 체크포인트든 어떤 환경에 넣어도 평가할 수 있습니다.

### 학습 환경

| Task ID | 내용 |
|---|---|
| `Isaac-Ant-DR-v0` | 도메인 랜덤화: 마찰 0.3~1.2, 몸통 질량 ×0.7~1.5, 무게중심 ±5 cm, 출발 방향 ±180°, 초기 속도 |
| `Isaac-Ant-DR-Rough-v0` | DR + 우리 지형 (높이 ≤ 5 cm, 진행 방향 200 m) |
| `Isaac-Ant-DR-Rough-PushNoise-v0` | + 밀기 (3~6초마다 ±1.0 m/s) + 관측 노이즈 |
| `Isaac-Ant-DR-Rough-Smooth-v0` | + 흔들림 패널티 (ang_vel_xy −0.2, lin_vel_z −0.75, action_rate −0.1) |
| `Isaac-Ant-DR-Rough-PushNoise-Smooth-v0` | 위 요소 전부 |
| `Isaac-Ant-DR-TM-v0` 외 3개 (`-PushNoise`, `-Smooth`, `-PushNoise-Smooth`) | 위와 같고, 지형만 팀원 학습 지형으로 바꾼 버전 (아래 참고) |

### 평가 전용 환경

| Task ID | 바꾼 조건 |
|---|---|
| `Isaac-Ant-Test-LowFriction-v0` | 바닥 마찰 0.25 / 0.2 |
| `Isaac-Ant-Test-Heavy-v0` | 몸통 ×1.8 |
| `Isaac-Ant-Test-WeakMotor-v0` | 모터 토크 ×2/3 |
| `Isaac-Ant-Test-Rough-v0` | 우리 검증 지형 (≤ 8 cm) |
| `Isaac-Ant-Test-Push-v0` | 2~4초마다 ±1.5 m/s 밀기 |
| `Isaac-Ant-Test-Heading-v0` | 출발 방향 ±180° |
| `Isaac-Ant-Test-Combined-v0` | 마찰 0.5 + 몸통 ×1.4 + 모터 ×0.85 + 검증 지형 + 밀기 + 방향 ±90° |
| `Isaac-Ant-Test-TM-v0` | 팀원 "처음 보는 환경"을 재구성한 것 (아래 참고) |

## 3. 팀원 지형 재구성 (TM)

팀원 코드가 아니라 발표 자료 7·8번 슬라이드를 보고 만든 근사치입니다. 실제 설정과 다른 부분은 직접 고쳐서 쓰면 됩니다. 정의는 `ant_robust_env_cfg.py`의 `TM_TRAIN_TERRAIN`, `TM_TEST_TERRAIN`, `AntTestTMEnvCfg`에 있습니다.

- **학습 지형**
  - 8 m × 8 m 타일, 진행 방향으로 20단계 (`curriculum=True`, 모두 첫 줄에서 출발)
  - 열 구성: 평지 3 · 요철 ≤ 6 cm 2 · 경사 1 · 계단 ≤ 10 cm 2 · 박스 ≤ 8 cm 2
- **검증 지형**
  - 지형 seed 24
  - 열 구성: 파도 4 · 장애물 3 · 거친 요철 ≤ 10 cm 3
  - 물성: body마다 마찰 0.3~0.7, 전체 질량 ×0.7~1.3, 밀기 없음
- **넘어짐 판정:** 학습 중에만 지형 기준 높이로 판정합니다(몸통에서 아래로 쏘는 레이캐스터 사용). 평가는 원래 판정 그대로입니다.
  - 레이캐스터를 쓰기 위해 TM 학습 환경은 `clone_in_fabric=False`로 설정했습니다. 그래서 학습 시작이 조금 느립니다.
- **재구성 정확도:** 강의 Baseline(4096 × 1000, seed 42) 모델을 `Isaac-Ant-Test-TM-v0`에서 평가하면 **39.5 ± 17.3**입니다. 원래 발표 자료의 값은 36.3 ± 18.3입니다.

## 4. 체크포인트

`checkpoints/<조건>/model_999.pt`에 있습니다. 모두 4096 envs × 1000 iter, seed 42로 학습했고, 학습 당시 설정은 각 폴더의 `params/env.yaml`, `params/agent.yaml`에 있습니다.

| 폴더 | 학습 Task ID |
|---|---|
| `dr` | Isaac-Ant-DR-v0 |
| `dr_rough` / `dr_rough_pushnoise` / `dr_rough_smooth` / `dr_rough_pushnoise_smooth` | Isaac-Ant-DR-Rough-... |
| `dr_tm` / `dr_tm_pushnoise` / `dr_tm_smooth` / `dr_tm_pushnoise_smooth` | Isaac-Ant-DR-TM-... |

## 5. 실행 예시

모두 `cd ~/IsaacLab_RS`에서 실행합니다.

```bash
# 평가: 원래 환경이나 팀원 검증 환경에 체크포인트 넣기 (강의 규칙: 100 envs, seed 24)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
    --task Isaac-Ant-v0 --headless --seed 24 --num_envs 100 --checkpoint <경로>/checkpoints/dr_tm/model_999.pt

# 팀원 검증 환경을 우리 재구성 대신 팀원 환경으로 쓰려면, --task만 팀원 Task ID로 바꾸면 됩니다.

# 학습
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Ant-DR-TM-v0 --headless --seed 42 --num_envs 4096 --max_iterations 1000 --run_name dr_tm_s42
```
