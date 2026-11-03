"""CarRacing-v3 환경 생성 및 전처리 래퍼.

래퍼 순서 (안쪽 -> 바깥쪽):
    CarRacing-v3
    -> HideScoreText             [수정 금지] 화면 아래 점수 숫자를 그리지 않음 (컴퓨터마다 다르게 그려짐)
    -> FrameCounter              [수정 금지] 원본 env.step 호출 수 = 샘플 예산
    -> RecordEpisodeStatistics   [수정 금지] 원본 보상 기준 에피소드 return (채점 기준)
    -> (학습 시) 보상 shaping     [수정 가능] 채점에는 영향 없음
    -> FrameSkip / Grayscale / FrameStack  [수정 가능] 관측·행동 전처리

바깥쪽 전처리는 자유롭게 바꿔도 되지만, make_env()의 [수정 금지] 부분은 건드리지 마세요.
평가(evaluate.py)도 같은 make_env(train=False)를 사용합니다.

제출(ONNX)과의 관계:
    공식 채점기(check_submission.py)는 이 파일을 쓰지 않고, 원본 RGB 프레임을 정책에 직접 넣습니다.
    그래서 관측 전처리를 바꾸면 아래 preprocess_torch()도 똑같이 바꿔야 합니다.
    (export.py가 두 결과가 일치하는지 자동으로 검사합니다.)
"""

from collections import deque

import gymnasium as gym
import numpy as np
import pygame
import torch
import torch.nn.functional as F

FRAME_SKIP = 4  # 같은 행동을 반복할 프레임 수
N_STACK = 4     # 쌓을 관측 수
GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


class BudgetExceeded(RuntimeError):
    pass


class FrameCounter(gym.Wrapper):
    """원본 환경의 step 호출 수를 센다. budget을 넘기면 예외를 던진다. [수정 금지]"""

    def __init__(self, env, budget=None, start=0):
        super().__init__(env)
        self.frames = start
        self.budget = budget

    def step(self, action):
        if self.budget is not None and self.frames >= self.budget:
            raise BudgetExceeded(f"sample budget {self.budget} exceeded")
        self.frames += 1
        return self.env.step(action)


class HideScoreText(gym.Wrapper):
    """화면 아래의 점수 숫자를 그리지 않는다. [수정 금지]

    숫자 글꼴은 pygame 버전과 운영체제마다 다르게 그려져서, 같은 정책도 컴퓨터마다 점수가 크게 달라지는
    원인이었다. 숫자를 빼면 트랙, 차, 계기판 막대는 어느 컴퓨터에서나 같은 화면이 된다.
    학습 환경(envs.py)과 공식 채점기(check_submission.py)가 같은 코드를 쓴다."""

    class _NoFont:
        def __init__(self, *args, **kwargs):
            pass

        def render(self, *args, **kwargs):
            return pygame.Surface((0, 0))

    def _without_font(self, fn, *args, **kwargs):
        original = pygame.font.Font
        pygame.font.Font = self._NoFont
        try:
            return fn(*args, **kwargs)
        finally:
            pygame.font.Font = original

    def reset(self, **kwargs):
        return self._without_font(self.env.reset, **kwargs)

    def step(self, action):
        return self._without_font(self.env.step, action)

    def render(self):
        return self._without_font(self.env.render)


class FrameSkip(gym.Wrapper):
    """같은 행동을 k 프레임 반복하고 보상을 합산한다."""

    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.k):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class Grayscale(gym.ObservationWrapper):
    """(96, 96, 3) RGB -> (96, 96) uint8 흑백."""

    def __init__(self, env):
        super().__init__(env)
        h, w, _ = env.observation_space.shape
        self.observation_space = gym.spaces.Box(0, 255, (h, w), np.uint8)

    def observation(self, obs):
        # preprocess_torch()와 똑같은 계산 순서 (행렬곱 @를 쓰면 반올림이 달라 일부 픽셀이 1씩 어긋남)
        gray = (obs.astype(np.float32) * GRAY_WEIGHTS).sum(axis=-1)
        return gray.astype(np.uint8)


class Resize84(gym.ObservationWrapper):
    """E10-E13: grayscale 96x96 -> 84x84; export uses matching torch interpolate."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0,255,(84,84),np.uint8)
    def observation(self, obs):
        x=torch.from_numpy(obs.copy()).unsqueeze(0).unsqueeze(0).float()
        return F.interpolate(x,size=(84,84),mode="bilinear",align_corners=False)[0,0].to(torch.uint8).numpy()


class FrameStack(gym.Wrapper):
    """최근 k 프레임을 쌓아 (k, H, W) uint8 관측을 만든다."""

    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        h, w = env.observation_space.shape
        self.observation_space = gym.spaces.Box(0, 255, (k, h, w), np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return np.stack(self.frames), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return np.stack(self.frames), reward, terminated, truncated, info


def make_env(continuous=False, train=True, budget=None, frames_start=0, render_mode=None):
    """
    continuous=False: 이산 행동 5개 (0: 없음, 1: 우회전, 2: 좌회전, 3: 가속, 4: 브레이크)  -> DQN, A2C
    continuous=True : 연속 행동 [steer(-1~1), gas(0~1), brake(0~1)]                      -> A2C 등
    train=True일 때만 보상 shaping 등 학습용 래퍼를 적용한다.
    """
    # ---------------- [수정 금지] ----------------
    env = HideScoreText(gym.make("CarRacing-v3", continuous=continuous, render_mode=render_mode))
    env = FrameCounter(env, budget=budget, start=frames_start)
    env = gym.wrappers.RecordEpisodeStatistics(env)  # info["episode"]["r"] = 채점용 원본 return
    # ---------------------------------------------

    # ---------------- [수정 가능] ----------------
    if train:
        pass  # TODO(개선): 보상 shaping, 보상 clipping 등은 여기서 (채점에는 영향 없음)
    env = FrameSkip(env, k=FRAME_SKIP)
    env = Grayscale(env)
    env = Resize84(env)
    env = FrameStack(env, k=N_STACK)
    # ---------------------------------------------
    return env


def preprocess_torch(frames):
    """ONNX 제출용: 위 관측 전처리(Grayscale + FrameStack)를 torch로 똑같이 구현.

    frames: (B, N_STACK, 96, 96, 3) uint8  — 최근 N_STACK개 agent step의 원본 RGB 프레임
    반환 : 에이전트가 학습 때 받던 관측과 같은 텐서, 여기서는 (B, N_STACK, 96, 96) uint8
    관측 전처리를 바꿨다면 이 함수도 같이 바꾸세요.
    """
    gray = (frames.float() * torch.from_numpy(GRAY_WEIGHTS)).sum(dim=-1)
    gray = gray.floor().reshape(-1,1,96,96)
    gray = F.interpolate(gray,size=(84,84),mode="bilinear",align_corners=False)
    return gray.reshape(frames.shape[0],frames.shape[1],84,84).to(torch.uint8)


def get_frame_counter(env):
    """래퍼 체인에서 FrameCounter를 찾는다."""
    while not isinstance(env, FrameCounter):
        env = env.env
    return env
