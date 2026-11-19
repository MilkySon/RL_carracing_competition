# CarRacing-v3 RL Competition — Skeleton

Gymnasium `CarRacing-v3`에서 강화학습 에이전트를 직접 만들고 개선하는 프로젝트입니다.
기본으로 **DQN**과 **A2C**(이산/연속) baseline이 들어 있고, 같은 인터페이스로 다른 알고리즘을 추가할 수 있습니다.

## 대회 규칙 (요약)

| 항목 | 내용 |
|---|---|
| 환경 | `CarRacing-v3` (이산 5행동 또는 연속 3차원 중 선택) |
| 샘플 예산 | 원본 환경 **300,000 프레임** (`env.step` 호출 수, frame skip 포함 전부) |
| 점수 | **공개되지 않은 트랙(seed)** 들에서의 원본 보상 return 평균 |
| 허용 | 알고리즘, 네트워크, 하이퍼파라미터, 관측/행동 전처리, 학습용 보상 shaping 등 모두 자유 |
| 금지 | 외부 데이터·사전학습 가중치 사용, `envs.py`의 `[수정 금지]` 영역 및 채점 로직 변경, 환경 내부 상태 접근 |

- 예산은 `FrameCounter`가 자동으로 세고, 초과하면 `BudgetExceeded`로 학습이 멈춥니다.
- 학습 중 검증(eval)은 별도 환경이라 예산에 포함되지 않습니다.
- seed마다 트랙이 새로 생성되므로 **처음 보는 트랙에서 잘 달리는 일반화 성능**이 중요합니다.

## 파일 구조

```
envs.py              환경 생성 + 전처리 래퍼 (frame skip, 흑백, frame stack)
networks.py          CNNEncoder / MLPHead / QNetwork  (Actor-Critic도 이 부품을 재사용)
buffers.py           ReplayBuffer (off-policy용)
agents/
  base.py            Agent 인터페이스: act / step / state_dict / load_state_dict
  dqn.py             DQN baseline (동작함)
  actor_critic.py    A2C baseline, 이산/연속 모두 지원 (동작함)
  __init__.py        에이전트 자동 등록 (agents/에 파일만 추가하면 됨)
train.py             공통 학습 루프 (체크포인트, resume, 로그)
evaluate.py          체크포인트 평가 (영상 저장 가능)
export.py            체크포인트 -> ONNX 제출 파일 변환 + 일치 검증
check_submission.py  공식 채점기 [수정 금지] — 제출 전 자체 점검용 (채점 서버도 같은 파일 사용)
colab_quickstart.ipynb   Colab용 노트북 (학습 -> 평가 -> 제출 파일)
local_quickstart.ipynb   내 PC(Windows 등)용 노트북 (VS Code, Jupyter)
```

## 빠른 시작

```bash
pip install swig
pip install -r requirements.txt

python train.py --agent dqn --seed 0                           # 학습 -> runs/dqn_s0/
python train.py --agent a2c --seed 0                           # A2C (이산) -> runs/a2c_s0/
python train.py --agent a2c --cfg continuous=True              # A2C (연속 행동)
python train.py --agent dqn --cfg lr=3e-4 batch_size=128       # 하이퍼파라미터 변경
python train.py --agent dqn --run-dir runs/dqn_s0 --resume      # 이어서 학습
python evaluate.py --ckpt runs/dqn_s0/best.pt                   # 공개 seed 1000-1009로 평가
python evaluate.py --ckpt runs/dqn_s0/best.pt --video videos/   # 주행 영상 저장
```

노트북으로 하려면 Colab에서는 `colab_quickstart.ipynb`, 내 PC에서는 `local_quickstart.ipynb`를 쓰세요. 로컬 노트북은 이 폴더에서 열고 프로젝트용 Python 환경을 커널로 고르면 됩니다. GPU가 없어도 학습은 되지만 CPU만 쓰면 약 2배 느립니다.

Colab에서는 `--run-dir`를 Google Drive에 두면 세션이 끊겨도 `--resume`으로 이어갈 수 있습니다.
(resume 시 replay buffer는 복원되지 않고 `learning_starts`만큼 다시 채운 뒤 학습을 재개합니다. 사용한 프레임 수는 이어서 계산됩니다.)

## 환경 정보

- 관측(전처리 후): `(4, 96, 96)` uint8 — 흑백 4프레임 stack, frame skip 4
- 이산 행동 (`continuous=False`): `0` 없음, `1` 우회전, `2` 좌회전, `3` 가속, `4` 브레이크
  - 세기가 고정됨: 조향 0.6, 가속 0.2, 브레이크 0.8. 조향과 가속을 동시에 할 수 없음
- 연속 행동 (`continuous=True`, CarRacing 원래 행동 공간): `[steer ∈ [-1,1], gas ∈ [0,1], brake ∈ [0,1]]`
- 두 방식 모두 보상·트랙·종료 조건은 같으므로 같은 기준으로 순위를 매깁니다. 어느 쪽을 쓸지도 전략입니다.
- 보상: 프레임당 −0.1, 새 타일 방문 시 +1000/N (N = 트랙 타일 수). 트랙 밖으로 크게 벗어나면 −100 후 종료
- 에피소드: 최대 1000 프레임 (= agent step 250). 900점 이상이면 "해결" 수준

## Baseline 참고 점수

30만 프레임으로 학습하고, 학습에 쓰지 않은 트랙 20개(seed 1000–1019)에서 `check_submission.py`로 평가한 `best.pt`의 결과입니다(DQN은 학습 seed 3개, A2C는 1개).

| baseline | 평균 점수 | 비고 |
|---|---|---|
| DQN (기본값) | 약 590~810 (seed 3개 평균 약 710) | 트랙별 편차가 큼 (최저 약 340~450, 최고 약 880~920) |
| A2C (이산/연속) | 약 −40 ~ −20 | **의도적으로 약한 baseline.** 환경 하나로 on-policy 학습을 하면 이 예산에서는 거의 학습되지 않습니다. PPO, 병렬 환경, GAE 등으로 개선해 보세요 |

학습 seed에 따라 점수가 ±100점 이상 달라질 수 있고, 같은 seed로 다시 학습해도 GPU 계산의 미세한 차이 때문에 결과가 달라집니다.

## 등급 기준

최종 점수(비공개 트랙 평균) 기준입니다.

| 등급 | 기준 |
|---|---|
| S | 850점 이상 (이 안에서 순위 경쟁) |
| A | 800점 이상 |
| B | 750점 이상 |
| C | 750점 미만 |

## 새 알고리즘 추가하기

학습 루프는 알고리즘과 무관하게 아래만 호출합니다.

```python
action  = agent.act(obs)                                          # 행동 선택
metrics = agent.step(obs, action, reward, next_obs, terminated, truncated)  # 저장 + 학습
```

데이터를 어떻게 저장하고 언제 학습할지는 **에이전트가 `step()` 안에서 결정**합니다.

| 방식 | `step()`에서 할 일 | 참고 |
|---|---|---|
| Off-policy (DQN, SAC, TD3) | `ReplayBuffer`에 추가 -> 매 스텝 샘플링해 업데이트 | `agents/dqn.py` |
| On-policy (A2C, PPO) | rollout 리스트에 추가 -> `n_steps`가 차면 업데이트 후 비움 | `agents/actor_critic.py` |

순서:
1. `Agent`를 상속한 클래스를 만들고 `act`, `step`, `state_dict`, `load_state_dict`, `export_policy`(제출용)를 구현합니다.
   가장 가까운 baseline(`dqn.py` 또는 `actor_critic.py`)을 복사해서 고치는 것이 빠릅니다.
2. 파일을 `agents/` 폴더에 넣으면 **자동으로 등록**됩니다. 클래스에 고유한 `name = "my_agent"`를 지정하고 `python train.py --agent my_agent`로 실행하세요. (상속한 클래스는 부모의 `name`을 물려받으니 꼭 새로 지정하세요.)
3. config의 `"continuous"` 값(기본 False)에 따라 이산/연속 행동 환경이 자동 선택되고, 에이전트 안에서는 `self.continuous`로 확인할 수 있습니다.

A2C(`actor_critic.py`)는 정책 분포가 이산이면 Categorical, 연속이면 Normal로 자동 선택됩니다.
`dist.log_prob(u)`와 `dist.entropy()`가 두 경우 모두 샘플당 스칼라이므로 **같은 update 코드로 둘 다 동작**합니다.
PPO로 확장할 때도 이 구조를 그대로 쓸 수 있습니다 (`act()`에서 old log_prob 저장 -> `update()`에서 여러 epoch + clip).

주의: `truncated`(시간 제한)일 때는 bootstrap해야 하고, `terminated`일 때만 다음 상태 가치를 0으로 둡니다. `ReplayBuffer`와 A2C의 return 계산은 이미 이렇게 처리합니다.

## 개선 아이디어 (출발점)

- DQN 계열: Double DQN, Dueling, n-step return, Prioritized Replay, 분포형 Q
- 표현 학습: data augmentation (random shift 등), 더 나은 encoder
- 전처리: 초반 줌인 애니메이션 프레임 건너뛰기, 관측 crop/resize, frame skip 조정, 행동 집합 재설계
- 탐험: epsilon 스케줄, noisy network
- 메모리: 프레임 단위 replay buffer로 버퍼 크기 확대
- 체크포인트 선택: 학습 중 검증 트랙이 5개뿐이라 `best.pt`가 꼭 최선은 아닙니다. `final.pt`와 비교하거나 검증 트랙을 늘려 보세요 (baseline에서도 `final.pt`가 100점 높은 경우가 있었습니다)
- Actor-Critic 계열: GAE, advantage 정규화, A2C -> PPO, SAC. 연속 행동 처리 개선 (tanh squashing의 log-prob 보정 등)

## 리더보드 제출 (ONNX)

```bash
python export.py --ckpt runs/dqn_s0/best.pt --out MyTeam.onnx   # 파일 이름은 반드시 팀 이름으로 시작
python check_submission.py --onnx MyTeam.onnx                   # 공식 채점기로 자체 점검
```

- **파일 이름은 팀 이름으로 시작**해야 합니다: `MyTeam.onnx`, `MyTeam_v2.onnx`, `MyTeam-final.onnx` 등.
  팀 이름 바로 뒤에는 `_` `-` 공백 `.` `(` `[` 중 하나가 와야 하고, 대소문자는 구분하지 않습니다. 팀원 누가 제출해도 같은 팀으로 집계됩니다.
- 팀은 **최대 2명**입니다. 팀 이름은 자유롭게 정하되(리더보드에 공개, 이름·학번은 비공개) `\ / : * ? " < > |`는 쓸 수 없습니다.
  **첫 제출 이후에는 팀 이름을 바꾸지 마세요.** 이전 채점 기록과 연결이 끊깁니다.
- 제출 파일은 정책만 담은 ONNX입니다. 채점 서버는 여러분의 코드를 실행하지 않고 `check_submission.py`로 이 파일만 돌립니다.
- 채점기는 **원본 RGB 프레임**을 정책에 넣습니다. 관측 전처리(`envs.py`)를 바꿨다면 `envs.preprocess_torch()`도 똑같이 바꾸세요.
  `export.py`가 학습 환경과 제출 파일의 행동 일치율을 보여 주니, **95% 미만이면 제출하지 마세요.**
  95~100% 사이는 정상입니다. Q값이나 확률이 거의 같은 행동 사이에서 미세한 계산 오차로 선택이 갈릴 수 있습니다.
- 행동 집합을 재설계했다면 `export_policy()`가 **환경에 들어가는 최종 행동**을 출력해야 합니다.
- **자체 점검 점수와 리더보드 점수:** 같은 파일이라도 컴퓨터(운영체제, CPU)에 따라 물리 계산이 아주 미세하게 달라서, 일부 트랙에서는 주행이 갈라져 **트랙별 점수가 크게 다를 수 있습니다.** 평균은 대체로 몇 점 이내로 맞습니다.
  트랙별 점수보다 평균을 보고, 공식 점수는 리더보드 기준입니다. 트랙 가장자리를 아슬아슬하게 도는 정책일수록 이런 차이에 흔들리므로, 안정적으로 주행하는 정책이 유리합니다.
- 채점: 매일 새벽 일괄 채점 후 **오전 10시에 리더보드 갱신**. **팀당** 하루에 가장 최근 제출물 1개만 채점됩니다.
- 공개 리더보드 점수는 비공개 트랙들로 계산하고, **최종 순위는 또 다른 비공개 트랙으로** 다시 매깁니다 (공개 점수가 가장 높은 제출물 기준).

## 최종 제출물

아래를 **zip 파일 하나**로 제출합니다.

1. 코드 전체 (이 저장소 구조 유지)
2. 최종 ONNX 파일과 그 원본 `best.pt`
3. 해당 run 폴더의 `config.json`, `train.csv`, `eval.csv` (예산 준수 확인용, `best.pt`와 같은 폴더에)

**재현 검증:** 모든 제출물은 자동 검증을 거치며, 상위권과 무작위로 뽑힌 팀은 교수자가 제출 코드와 설정으로 **같은 예산 안에서 다시 학습**해 점수가 재현되는지 확인합니다. 재현되지 않거나 `[수정 금지]` 코드를 고친 경우 최종 순위에서 제외될 수 있습니다.
- 학습은 `python train.py --agent <이름> --seed <seed> --cfg ...`로 재현할 수 있어야 합니다. `config.json`에 기록된 설정과 seed가 그대로 쓰입니다.
- `train.py`의 옵션을 바꿨거나 추가 패키지가 필요하다면, 정확한 재현 명령과 패키지 목록을 `REPRODUCE.txt` 파일에 적어 zip에 함께 넣으세요.
