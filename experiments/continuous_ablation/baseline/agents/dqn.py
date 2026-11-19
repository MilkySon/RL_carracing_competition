"""기본 DQN (Mnih et al., 2015).

의도적으로 단순하게 만든 baseline입니다. 개선 아이디어 예시:
    Double DQN, Dueling, PER, n-step return, 분포형 Q, data augmentation, 더 나은 탐험 ...
"""

import numpy as np
import torch
import torch.nn.functional as F

from buffers import ReplayBuffer
from networks import QNetwork

from .base import Agent


class _GreedyPolicy(torch.nn.Module):
    def __init__(self, q):
        super().__init__()
        self.q = q

    def forward(self, obs):
        return self.q(obs).argmax(dim=1)


class DQNAgent(Agent):
    name = "dqn"

    default_config = {
        "lr": 1e-4,
        "gamma": 0.99,
        "batch_size": 64,
        "buffer_size": 50_000,
        "learning_starts": 5_000,         # 버퍼에 이만큼 쌓인 뒤 학습 시작 (agent step 기준)
        "train_freq": 1,                  # 몇 agent step마다 업데이트할지
        "target_update_interval": 2_000,  # 몇 번 업데이트마다 target 동기화할지
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_decay_steps": 25_000,        # agent step 기준 (= 10만 프레임). 예산에 맞춰 조정하면 효과가 큼
        "max_grad_norm": 10.0,
    }

    def __init__(self, obs_shape, action_space, config, device):
        super().__init__(obs_shape, action_space, config, device)
        assert not self.continuous, "DQN은 이산 행동만 지원합니다 (continuous=False)."
        self.n_actions = action_space.n
        self.q = QNetwork(obs_shape, self.n_actions).to(device)
        self.q_target = QNetwork(obs_shape, self.n_actions).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optim = torch.optim.Adam(self.q.parameters(), lr=self.cfg["lr"])
        self.buffer = None  # 첫 step()에서 생성 (평가만 할 때는 메모리를 쓰지 않도록)
        self.steps = 0
        self.updates = 0

    # ---------------------------------------------------------------- 행동 선택
    def epsilon(self):
        c = self.cfg
        frac = min(1.0, self.steps / c["eps_decay_steps"])
        return c["eps_start"] + frac * (c["eps_end"] - c["eps_start"])

    def act(self, obs, deterministic=False):
        if not deterministic and np.random.rand() < self.epsilon():
            return int(np.random.randint(self.n_actions))
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, device=self.device).unsqueeze(0)
            return int(self.q(obs_t).argmax(dim=1).item())

    # ---------------------------------------------------------------- 저장 + 학습
    def step(self, obs, action, reward, next_obs, terminated, truncated):
        if self.buffer is None:
            self.buffer = ReplayBuffer(self.cfg["buffer_size"], self.obs_shape)
        self.buffer.add(obs, action, reward, next_obs, terminated)
        self.steps += 1

        c = self.cfg
        # 버퍼 크기로 판단하므로, 버퍼 없이 resume해도 learning_starts만큼 다시 채운 뒤 학습 재개
        if len(self.buffer) < c["learning_starts"] or self.steps % c["train_freq"] != 0:
            return {}
        metrics = self.update(self.buffer.sample(c["batch_size"], self.device))
        metrics["epsilon"] = self.epsilon()
        return metrics

    def update(self, batch):
        c = self.cfg
        q = self.q(batch["obs"]).gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.q_target(batch["next_obs"]).max(dim=1).values
            target = batch["rewards"] + c["gamma"] * (1.0 - batch["dones"]) * next_q
        loss = F.smooth_l1_loss(q, target)

        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), c["max_grad_norm"])
        self.optim.step()

        self.updates += 1
        if self.updates % c["target_update_interval"] == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return {"loss": loss.item(), "q_mean": q.mean().item()}

    # ---------------------------------------------------------------- 제출 (ONNX)
    def export_policy(self):
        return _GreedyPolicy(self.q)

    # ---------------------------------------------------------------- 체크포인트
    def state_dict(self):
        return {
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
            "optim": self.optim.state_dict(),
            "steps": self.steps,
            "updates": self.updates,
        }

    def load_state_dict(self, state):
        self.q.load_state_dict(state["q"])
        self.q_target.load_state_dict(state["q_target"])
        self.optim.load_state_dict(state["optim"])
        self.steps = state["steps"]
        self.updates = state["updates"]
