"""Replay buffer (off-policy용).

단순하게 obs와 next_obs를 모두 uint8로 저장합니다.
(4, 96, 96) 관측 기준 transition당 약 74KB -> capacity 50,000이면 약 3.7GB RAM.

TODO(개선): 프레임 단위로 저장하면 메모리를 약 1/8로 줄일 수 있고, 그만큼 버퍼를 키울 수 있습니다.
On-policy 방법(A2C/PPO)은 이 버퍼 대신 짧은 rollout 버퍼를 에이전트 안에 직접 두면 됩니다.
"""

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, obs_shape, action_shape=(), action_dtype=np.int64):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity, *action_shape), dtype=action_dtype)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)  # terminated만 1 (truncated는 bootstrap)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, terminated):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(terminated)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        to = lambda x: torch.as_tensor(x[idx], device=device)
        return {
            "obs": to(self.obs),
            "actions": to(self.actions),
            "rewards": to(self.rewards),
            "next_obs": to(self.next_obs),
            "dones": to(self.dones),
        }

    def __len__(self):
        return self.size
