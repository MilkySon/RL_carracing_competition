"""체크포인트 -> ONNX 제출 파일 변환.

    python export.py --ckpt runs/dqn_s0/best.pt --out MyTeam.onnx

1. agent.export_policy()를 원본 RGB 프레임 입력용으로 감싸 ONNX로 저장합니다.
   (관측 전처리는 envs.preprocess_torch()가 ONNX 안에 들어갑니다.)
2. 검증: 학습 때 쓰던 환경(make_env)의 에이전트 행동과, 공식 채점 방식의 ONNX 행동이
   같은지 한 에피소드 동안 비교합니다. 일치율이 낮으면 preprocess_torch()나
   export_policy()가 학습 때와 다르다는 뜻입니다.
3. 제출 전 최종 확인: python check_submission.py --onnx MyTeam.onnx
"""

import argparse
import warnings
from collections import deque

import gymnasium as gym
import numpy as np
import onnx
import torch

import check_submission
from envs import FRAME_SKIP, N_STACK, make_env, preprocess_torch
from evaluate import load_agent


class Submission(torch.nn.Module):
    def __init__(self, policy, continuous):
        super().__init__()
        self.policy = policy
        self.continuous = continuous

    def forward(self, frames):
        action = self.policy(preprocess_torch(frames))
        return action.float() if self.continuous else action.long()


def export(agent, out_path):
    model = Submission(agent.export_policy(), agent.continuous).eval()
    dummy = torch.zeros(1, N_STACK, 96, 96, 3, dtype=torch.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(model, (dummy,), out_path, input_names=["frames"],
                          output_names=["action"], opset_version=17, dynamo=False)
    m = onnx.load(out_path)
    for k, v in {"continuous": str(agent.continuous).lower(),
                 "frame_skip": str(FRAME_SKIP), "n_stack": str(N_STACK)}.items():
        m.metadata_props.add(key=k, value=v)
    onnx.save(m, out_path)


def verify(agent, out_path, seed=1000):
    """같은 seed, 같은 행동으로 두 환경을 나란히 돌리며 행동 일치율을 잰다."""
    sess, cfg = check_submission.load_policy(out_path)
    train_env = make_env(continuous=agent.continuous, train=False)
    raw_env = check_submission.HideScoreText(gym.make("CarRacing-v3", continuous=agent.continuous))
    obs, _ = train_env.reset(seed=seed)
    raw, _ = raw_env.reset(seed=seed)
    frames = deque([raw] * cfg["n_stack"], maxlen=cfg["n_stack"])

    match, total, done = 0, 0, False
    while not done:
        a_agent = agent.act(obs, deterministic=True)
        a_onnx = check_submission.to_env_action(
            sess.run(["action"], {"frames": np.stack(frames)[None]})[0], cfg, raw_env)
        match += bool(np.allclose(a_agent, a_onnx, atol=1e-3))
        total += 1
        obs, _, t1, tr1, _ = train_env.step(a_agent)
        for _ in range(cfg["frame_skip"]):  # 공식 채점기와 같은 방식으로 원본 환경 진행
            raw, _, t2, tr2, _ = raw_env.step(a_agent)
            if t2 or tr2:
                break
        frames.append(raw)
        done = t1 or tr1 or t2 or tr2
    return match / total, total


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True, help="제출 파일 이름: <팀이름>.onnx (예: MyTeam.onnx, MyTeam_v2.onnx)")
    args = p.parse_args()

    agent = load_agent(args.ckpt, torch.device("cpu"))
    export(agent, args.out)
    print(f"saved {args.out}  (continuous={agent.continuous}, frame_skip={FRAME_SKIP}, n_stack={N_STACK})")

    rate, n = verify(agent, args.out)
    print(f"행동 일치율: {rate * 100:.1f}% ({n} steps)")
    if rate < 0.95:
        print("경고: 학습 환경과 제출 파일의 행동이 다릅니다. preprocess_torch()와 export_policy()를 확인하세요.")
    elif rate < 1.0:
        print("참고: 95% 이상이면 정상입니다. Q값/확률이 거의 같은 행동 사이에서 미세한 계산 오차로 선택이 갈릴 수 있습니다.")
    print(f"제출 전 확인: python check_submission.py --onnx {args.out}")
