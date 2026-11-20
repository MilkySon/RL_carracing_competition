"""모든 에이전트가 따르는 공통 인터페이스.

train.py와 evaluate.py는 이 인터페이스만 사용합니다.
새 알고리즘을 만들 때는 이 클래스를 상속해 아래 메서드를 채우고, 고유한 name을 정해 agents/ 폴더에 파일로 두면 자동 등록됩니다.

핵심 설계: 데이터 저장과 학습 시점은 에이전트가 스스로 결정합니다 (step() 안에서).
    - DQN/SAC 같은 off-policy: step()에서 replay buffer에 넣고 매 스텝(또는 N 스텝마다) 업데이트
    - A2C/PPO 같은 on-policy : step()에서 rollout 버퍼에 쌓다가 n_steps가 차면 업데이트 후 비움
그래서 학습 루프(train.py)는 알고리즘이 바뀌어도 그대로 둘 수 있습니다.
"""

from abc import ABC, abstractmethod

import gymnasium as gym


class Agent(ABC):
    name = "base"

    # 하이퍼파라미터 기본값. train.py의 --cfg key=value로 덮어쓸 수 있음.
    # "continuous": True면 연속 행동 CarRacing, False(기본)면 이산 행동 5개.
    default_config = {}

    @classmethod
    def is_continuous(cls, config=None):
        """환경을 만들기 전에 train.py/evaluate.py가 행동 공간 종류를 알기 위해 호출."""
        return bool({**cls.default_config, **(config or {})}.get("continuous", False))

    def __init__(self, obs_shape, action_space, config, device):
        self.obs_shape = obs_shape
        self.action_space = action_space
        self.continuous = isinstance(action_space, gym.spaces.Box)
        unknown = set(config or {}) - set(self.default_config) - {"continuous"}
        if unknown:  # 오타 등으로 아무 효과 없는 설정을 알려 줌
            print(f"[경고] {self.name}에 없는 설정이라 무시됩니다: {', '.join(sorted(unknown))}  "
                  f"(가능한 설정: {', '.join(self.default_config)})")
        self.cfg = {**self.default_config, **(config or {})}
        self.device = device

    @abstractmethod
    def act(self, obs, deterministic=False):
        """obs: np.ndarray (C, H, W) uint8 -> 환경에 넣을 action.
        deterministic=True는 평가용 (탐험 없이)."""

    @abstractmethod
    def step(self, obs, action, reward, next_obs, terminated, truncated):
        """환경 한 스텝이 끝날 때마다 호출. 저장/학습을 수행하고 로깅할 지표 dict를 반환 (없으면 {})."""

    def export_policy(self):
        """제출용 결정적 정책 nn.Module을 반환 (export.py가 ONNX로 변환).

        입력 : 학습 때와 같은 관측 텐서 (B, C, H, W) uint8
        출력 : 환경에 들어가는 최종 행동
               이산 -> (B,) int64 행동 번호,  연속 -> (B, 3) float32 [steer, gas, brake]
        act(obs, deterministic=True)와 같은 행동을 내야 합니다.
        """
        raise NotImplementedError(f"{type(self).__name__}.export_policy()를 구현하세요.")

    @abstractmethod
    def state_dict(self):
        """체크포인트에 저장할 모든 상태 (네트워크, optimizer, 내부 카운터 등)."""

    @abstractmethod
    def load_state_dict(self, state):
        """state_dict()로 저장한 상태를 되살림. 체크포인트 저장/불러오기는 train.py와 evaluate.py가 담당."""
