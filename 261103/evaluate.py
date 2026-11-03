"""체크포인트(.pt) 평가 스크립트 (자체 점검, 영상 저장용).

    python evaluate.py --ckpt runs/dqn_s0/best.pt
    python evaluate.py --ckpt runs/dqn_s0/best.pt --seeds 1000-1019 --video videos/

점수 = 트랙(seed)별 원본 보상 return의 평균. (학습용 보상 shaping은 반영되지 않음)
리더보드 채점은 ONNX 제출 파일로 check_submission.py가 하며, 공개되지 않은 seed를 씁니다.
"""

import argparse
import json

import gymnasium as gym
import numpy as np
import torch

from agents import AGENTS
from envs import make_env

PUBLIC_EVAL_SEEDS = list(range(1000, 1010))  # 자체 점검용 공개 seed (채점용 seed는 비공개)


def evaluate(agent, seeds, video_dir=None):
    env = make_env(continuous=agent.continuous, train=False,
                   render_mode="rgb_array" if video_dir else None)
    if video_dir:
        # 영상 저장에는 moviepy가 필요. agent step마다 한 프레임씩 녹화되므로 fps = 50 / frame_skip(4)
        env = gym.wrappers.RecordVideo(env, video_dir, episode_trigger=lambda _: True, fps=12)

    returns = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(agent.act(obs, deterministic=True))
            done = terminated or truncated
        returns.append(float(info["episode"]["r"]))
    env.close()
    return returns


def load_agent(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    agent_cls = AGENTS[ckpt["agent"]]
    env = make_env(continuous=agent_cls.is_continuous(ckpt["config"]), train=False)
    agent = agent_cls(env.observation_space.shape, env.action_space, ckpt["config"], device)
    env.close()
    agent.load_state_dict(ckpt["state"])
    return agent


def parse_seeds(s):
    """"1000-1009" 또는 "1,2,3" -> seed 목록. (check_submission.py에도 같은 함수가 있음: 채점기는 단독으로 동작해야 해서)"""
    if "-" in s:
        a, b = map(int, s.split("-"))
        return list(range(a, b + 1))
    return [int(x) for x in s.split(",")]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--seeds", type=parse_seeds, default=PUBLIC_EVAL_SEEDS,
                   help="예: 1000-1009 또는 1,2,3")
    p.add_argument("--video", default=None, help="영상 저장 폴더 (선택)")
    p.add_argument("--out", default=None, help="결과 JSON 저장 경로 (선택)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = load_agent(args.ckpt, device)
    returns = evaluate(agent, args.seeds, args.video)

    for s, r in zip(args.seeds, returns):
        print(f"seed {s}: {r:8.1f}")
    print(f"SCORE (mean over {len(returns)} tracks): {np.mean(returns):.1f} ± {np.std(returns):.1f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"ckpt": args.ckpt, "seeds": args.seeds, "returns": returns,
                       "mean": float(np.mean(returns)), "std": float(np.std(returns))}, f, indent=2)
