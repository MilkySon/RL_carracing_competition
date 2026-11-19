"""신경망 구성 요소.

Encoder(이미지 -> 특징 벡터)와 Head(특징 -> 출력)를 분리해 두었습니다.
    DQN          : CNNEncoder + MLPHead(n_actions)                 -> Q(s, ·)
    Actor-Critic : CNNEncoder 하나를 actor head와 critic head가 공유 (encoder 분리는 확장 과제)
"""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """Nature-DQN 스타일 CNN. 입력: (B, C, H, W) uint8, 출력: (B, feature_dim)."""

    def __init__(self, obs_shape, feature_dim=512):
        super().__init__()
        c, h, w = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, c, h, w)).shape[1]
        self.fc = nn.Sequential(nn.Linear(n_flat, feature_dim), nn.ReLU())
        self.feature_dim = feature_dim

    def forward(self, obs):
        x = obs.float() / 255.0
        return self.fc(self.conv(x))


class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class QNetwork(nn.Module):
    """Q(s, ·): 이산 행동별 Q값."""

    def __init__(self, obs_shape, n_actions):
        super().__init__()
        self.encoder = CNNEncoder(obs_shape)
        self.head = MLPHead(self.encoder.feature_dim, n_actions)

    def forward(self, obs):
        return self.head(self.encoder(obs))
