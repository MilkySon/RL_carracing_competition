"""기본 A2C (Advantage Actor-Critic, n-step return). 이산/연속 행동 모두 지원.

의도적으로 단순하게 만든 baseline입니다. 개선 아이디어 예시:
    GAE(lambda), advantage 정규화, PPO (clipped objective, 여러 epoch), 병렬 환경, 학습률 스케줄 ...

DQN과 같은 Agent 인터페이스를 따르므로 train.py / evaluate.py는 수정할 필요가 없습니다.
    python train.py --agent a2c                        # 이산 5행동 (DQN과 같은 조건)
    python train.py --agent a2c --cfg continuous=True  # 연속 행동

행동 공간에 따라 정책 분포만 달라지고, 나머지(update 코드 포함)는 동일하게 쓸 수 있습니다.
    이산: Categorical(logits)                 -> action = 정수 0~4
    연속: Normal(mean, std) (3차원 독립)      -> 샘플 u를 to_env_action()으로 범위 변환
          action = [steer(-1~1), gas(0~1), brake(0~1)]
어느 경우든 dist.log_prob(u), dist.entropy()는 샘플마다 스칼라 하나를 돌려줍니다.

On-policy 구조:
    step()에서 transition을 rollout에 쌓다가 n_steps개가 모이면 update() 후 비움.
Off-policy(SAC 등)로 가려면 DQNAgent처럼 buffers.ReplayBuffer를 쓰면 됩니다.
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Independent, Normal

from networks import CNNEncoder, MLPHead

from .base import Agent


class ActorCriticNet(nn.Module):
    """공유 CNN encoder + actor head + critic head(V)."""

    def __init__(self, obs_shape, action_space):
        super().__init__()
        self.continuous = isinstance(action_space, gym.spaces.Box)
        n_out = action_space.shape[0] if self.continuous else action_space.n
        self.encoder = CNNEncoder(obs_shape)
        self.actor = MLPHead(self.encoder.feature_dim, n_out)   # 이산: logits, 연속: 평균
        self.critic = MLPHead(self.encoder.feature_dim, 1)
        if self.continuous:
            self.log_std = nn.Parameter(torch.zeros(n_out))

    def forward(self, obs):
        """-> (dist, value).  dist.sample() / dist.log_prob(u) / dist.entropy()"""
        z = self.encoder(obs)
        out = self.actor(z)
        value = self.critic(z).squeeze(-1)
        if self.continuous:
            dist = Independent(Normal(out, self.log_std.exp()), 1)
        else:
            dist = Categorical(logits=out)
        return dist, value


def to_env_action(u):
    """연속 행동용: 제약 없는 u (3,) -> CarRacing 행동 범위."""
    t = np.tanh(u)
    return np.array([t[0], (t[1] + 1) / 2, (t[2] + 1) / 2], dtype=np.float32)


class _DeterministicPolicy(nn.Module):
    """제출용: act(obs, deterministic=True)와 같은 행동 (분포 대신 평균/argmax)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, obs):
        out = self.net.actor(self.net.encoder(obs))
        if not self.net.continuous:
            return out.argmax(dim=-1)
        t = torch.tanh(out)  # to_env_action()의 torch 버전
        return torch.stack([t[:, 0], (t[:, 1] + 1) / 2, (t[:, 2] + 1) / 2], dim=-1)


class ActorCriticAgent(Agent):
    name = "a2c"

    default_config = {
        "continuous": False,   # True: 연속 행동 환경
        "lr": 3e-4,
        "gamma": 0.99,
        "n_steps": 128,        # 몇 스텝 모아서 한 번 업데이트할지
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "normalize_advantage": True,
        "reward_scale": 0.1,   # 학습용 보상 배율 (채점은 원본 보상 기준)
        "max_grad_norm": 0.5,
    }

    def __init__(self, obs_shape, action_space, config, device):
        super().__init__(obs_shape, action_space, config, device)
        self.net = ActorCriticNet(obs_shape, action_space).to(device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.cfg["lr"])
        self.rollout = []  # (obs, u, reward, next_obs, terminated, truncated)
        self.steps = 0
        self._last_u = None  # act()에서 샘플한 u. 학습(log_prob)에는 env action이 아니라 u를 사용

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, device=self.device).unsqueeze(0)
            dist, _ = self.net(obs_t)
            if deterministic:
                u = dist.mean if self.continuous else dist.probs.argmax(dim=-1)
            else:
                u = dist.sample()
        self._last_u = u.squeeze(0).cpu().numpy()
        return to_env_action(self._last_u) if self.continuous else int(self._last_u)

    def step(self, obs, action, reward, next_obs, terminated, truncated):
        self.rollout.append((obs, self._last_u, reward, next_obs, terminated, truncated))
        self.steps += 1
        if len(self.rollout) < self.cfg["n_steps"]:
            return {}
        metrics = self.update()
        self.rollout.clear()
        return metrics

    def update(self):
        c, dev = self.cfg, self.device
        obs, u, rewards, next_obs, term, trunc = zip(*self.rollout)

        # 1. rollout -> 텐서
        obs = torch.as_tensor(np.stack(obs), device=dev)
        next_obs = torch.as_tensor(np.stack(next_obs), device=dev)
        u = torch.as_tensor(np.stack(u), device=dev)  # 이산: int64 (n,), 연속: float32 (n, 3)
        # 보상 스케일링: return이 수백 단위면 value loss가 공유 encoder를 지배해 정책이 무너짐
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=dev) * c["reward_scale"]
        term = torch.as_tensor(term, dtype=torch.float32, device=dev)
        trunc = torch.as_tensor(trunc, dtype=torch.float32, device=dev)

        # 2. n-step return (뒤에서부터 누적)
        #    - rollout 마지막: 에피소드가 계속되므로 V(next_obs)로 bootstrap
        #    - truncated (시간 제한): 환경이 끝난 게 아니므로 V(next_obs)로 bootstrap
        #    - terminated: 진짜 종료이므로 미래 가치 0
        #    확장: GAE(lambda)
        with torch.no_grad():
            _, next_v = self.net(next_obs)
        returns = torch.zeros_like(rewards)
        g = next_v[-1]
        for t in reversed(range(len(rewards))):
            if trunc[t]:
                g = next_v[t]
            g = rewards[t] + c["gamma"] * (1.0 - term[t]) * g
            returns[t] = g

        # 3. loss
        dist, values = self.net(obs)
        advantage = returns - values
        adv = advantage.detach()
        if c["normalize_advantage"]:  # -100 같은 큰 보상에 정책이 한쪽으로 쏠리는 것을 완화
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        policy_loss = -(dist.log_prob(u) * adv).mean()
        value_loss = advantage.pow(2).mean()
        entropy = dist.entropy().mean()
        loss = policy_loss + c["value_coef"] * value_loss - c["entropy_coef"] * entropy

        # 4. optimizer step
        #    확장: PPO라면 여기서 여러 epoch / minibatch + clipped objective
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), c["max_grad_norm"])
        self.optim.step()
        return {"loss": loss.item(), "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(), "entropy": entropy.item()}

    def export_policy(self):
        return _DeterministicPolicy(self.net)

    def state_dict(self):
        return {"net": self.net.state_dict(), "optim": self.optim.state_dict(), "steps": self.steps}

    def load_state_dict(self, state):
        self.net.load_state_dict(state["net"])
        self.optim.load_state_dict(state["optim"])
        self.steps = state["steps"]
