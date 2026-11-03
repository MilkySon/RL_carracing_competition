"""학습 스크립트 (모든 에이전트 공통).

    python train.py --agent dqn --seed 0
    python train.py --agent dqn --cfg lr=3e-4 batch_size=128
    python train.py --agent dqn --run-dir runs/dqn_s0 --resume      # Colab 세션이 끊긴 뒤 이어서

샘플 예산(--budget)은 원본 환경의 step 호출 수(프레임)입니다. frame skip을 바꿔도 공정하게 비교됩니다.
검증용 평가(eval)는 별도 환경에서 돌기 때문에 예산에 포함되지 않습니다.

출력 (run-dir):
    config.json   실행 설정
    train.csv     학습 에피소드별 return (원본 보상 기준)
    eval.csv      주기적 검증 점수
    latest.pt     최신 체크포인트 (resume용, 검증할 때마다 저장)
    best.pt       검증 점수 최고 체크포인트 (제출용)
    final.pt      예산을 다 쓴 마지막 시점의 체크포인트

--resume으로 이어서 학습하면 에이전트 종류, 설정(--cfg), seed는 체크포인트에 저장된 값을 씁니다.
"""

import argparse
import ast
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

from agents import AGENTS
from direction_utils import direction_label
from envs import BudgetExceeded, get_frame_counter, make_env
from evaluate import evaluate

DEFAULT_BUDGET = 300_000            # 대회 샘플 예산 (원본 프레임 수)
VALIDATION_SEEDS = list(range(500, 505))  # 학습 중 검증용 트랙


def train_track(seed, episode):
    """학습용 트랙 번호. 검증 트랙(500-504), 공개 점검 트랙(1000-1009)과 겹치지 않고 seed마다 다른 트랙을 쓴다."""
    return 100_000 * (seed + 1) + episode


def parse_cfg(items):
    cfg = {}
    for item in items:
        if "=" not in item:
            sys.exit(f"--cfg 항목은 key=value 형식이어야 합니다: {item!r}  (예: --cfg lr=3e-4 batch_size=128)")
        key, value = item.split("=", 1)
        if value.lower() in ("true", "false", "none"):  # 소문자 true/false도 허용 (문자열 "false"는 참으로 취급되므로)
            cfg[key] = {"true": True, "false": False, "none": None}[value.lower()]
            continue
        try:
            cfg[key] = ast.literal_eval(value)  # 숫자, True/False, 리스트 등
        except (ValueError, SyntaxError):
            cfg[key] = value  # 그 밖에는 문자열 그대로
    return cfg


def save_checkpoint(path, agent, seed, frames, episodes, best_score):
    torch.save({"agent": agent.name, "config": agent.cfg, "seed": seed, "state": agent.state_dict(),
                "frames": frames, "episodes": episodes, "best_score": best_score}, path)


def append_csv(path, row):
    """CSV에 한 줄 추가. 새 열이 생기면(예: 학습 시작 후 loss 등) 헤더를 넓혀 파일 전체를 다시 쓴다."""
    header = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            header = next(csv.reader(f), [])
    new_keys = [k for k in row if k not in header]
    if header and not new_keys:
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=header).writerow(row)
        return
    fields = header + new_keys
    rows = []
    if header:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    rows.append(row)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def truncate_csv(path, max_frames):
    """resume할 때: 체크포인트 이후에 기록된 줄을 지운다 (그 구간은 다시 학습하므로)."""
    if not os.path.exists(path):
        return
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, [r for r in reader if int(r["frames"]) <= max_frames]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="dqn", choices=list(AGENTS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--cfg", nargs="*", default=[], help="하이퍼파라미터 덮어쓰기: key=value ...")
    p.add_argument("--eval-every", type=int, default=50_000, help="몇 프레임마다 검증할지")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    run_dir = args.run_dir or f"runs/{args.agent}_s{args.seed}"
    os.makedirs(run_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent_name, seed = args.agent, args.seed
    frames, episodes = 0, 0
    ckpt = None
    latest_path = os.path.join(run_dir, "latest.pt")
    # 학습 폴더 보호: 이전 기록과 새 학습이 섞이지 않도록
    previous = [f for f in ("train.csv", "eval.csv", "latest.pt", "best.pt") if os.path.exists(os.path.join(run_dir, f))]
    if args.resume and previous and set(previous) <= {"train.csv", "eval.csv"}:
        # 첫 체크포인트(latest.pt)가 저장되기 전에 멈춘 기록: 이어갈 상태가 없으니 로그를 비우고 처음부터
        print("참고: 체크포인트가 저장되기 전에 멈춘 기록이라 처음부터 다시 학습합니다.")
        truncate_csv(os.path.join(run_dir, "train.csv"), 0)
        truncate_csv(os.path.join(run_dir, "eval.csv"), 0)
    elif previous and not (args.resume and os.path.exists(latest_path)):
        sys.exit(f"{run_dir}에 이전 학습 기록이 있습니다 ({', '.join(previous)}).\n"
                 f"  - 이어서 학습하려면 --resume 을 붙이세요 (폴더에 latest.pt가 있어야 합니다).\n"
                 f"  - 새로 학습하려면 다른 --run-dir 을 쓰세요 (예: --run-dir {run_dir}_2).")
    if args.resume and os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        frames, episodes = ckpt["frames"], ckpt["episodes"]
        print(f"resume from {latest_path} (frames={frames}, episodes={episodes})")
        if frames >= args.budget:
            print(f"이미 {frames:,}프레임까지 학습된 체크포인트입니다 (예산 {args.budget:,}프레임). 추가 학습 없이 종료합니다.\n"
                  f"  새로 학습하려면 다른 --run-dir 을 쓰세요 (예: --run-dir {run_dir}_2).")
            return
        # 에이전트 종류, seed, 설정은 체크포인트를 따른다
        if ckpt["agent"] not in AGENTS:
            sys.exit(f"체크포인트의 에이전트 '{ckpt['agent']}'를 agents/ 폴더에서 찾을 수 없습니다.")
        if ckpt["agent"] != args.agent:
            print(f"참고: 체크포인트의 에이전트 '{ckpt['agent']}'로 이어서 학습합니다 (--agent {args.agent} 무시).")
        if args.cfg:
            print("참고: --resume에서는 체크포인트에 저장된 설정을 씁니다 (--cfg 무시).")
        agent_name, seed = ckpt["agent"], ckpt.get("seed", args.seed)
        # 체크포인트 이후에 기록된 로그는 지움 (그 구간을 다시 학습하므로 기록이 겹치지 않게)
        truncate_csv(os.path.join(run_dir, "train.csv"), frames)
        truncate_csv(os.path.join(run_dir, "eval.csv"), frames)

    np.random.seed(seed)
    torch.manual_seed(seed)
    agent_cls = AGENTS[agent_name]
    config = ckpt["config"] if ckpt else parse_cfg(args.cfg)
    env = make_env(continuous=agent_cls.is_continuous(config), train=True,
                   budget=args.budget, frames_start=frames)
    counter = get_frame_counter(env)
    agent = agent_cls(env.observation_space.shape, env.action_space, config, device)
    if ckpt:
        agent.load_state_dict(ckpt["state"])
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"agent": agent_name, "seed": seed, "budget": args.budget,
                   "config": agent.cfg}, f, indent=2)

    print(f"agent={agent_name} action={'continuous' if agent.continuous else 'discrete'} "
          f"device={device} budget={args.budget:,} frames")
    best_score = ckpt.get("best_score", -np.inf) if ckpt else -np.inf
    start_frames = frames
    next_eval = (frames // args.eval_every + 1) * args.eval_every
    metrics = defaultdict(list)
    t0 = time.time()
    last_eval = frames

    def run_eval():
        nonlocal best_score, last_eval
        last_eval = counter.frames
        returns = evaluate(agent, VALIDATION_SEEDS)
        score = float(np.mean(returns))
        append_csv(os.path.join(run_dir, "eval.csv"),
                   {"frames": counter.frames, "score": score, "std": float(np.std(returns))})
        print(f"  >> eval @ {counter.frames:,}: {score:.1f} ± {np.std(returns):.1f}")
        if score > best_score:
            best_score = score
            save_checkpoint(os.path.join(run_dir, "best.pt"), agent, seed, counter.frames, episodes, best_score)
        save_checkpoint(latest_path, agent, seed, counter.frames, episodes, best_score)

    obs, _ = env.reset(seed=train_track(seed, episodes))

    try:
        while counter.frames < args.budget:
            action = agent.act(obs)
            if hasattr(agent, "add_direction_example"):
                agent.add_direction_example(obs, direction_label(env))
            next_obs, reward, terminated, truncated, info = env.step(action)
            for k, v in agent.step(obs, action, reward, next_obs, terminated, truncated).items():
                metrics[k].append(v)
            obs = next_obs

            if terminated or truncated:
                episodes += 1
                ep = info["episode"]
                row = {"frames": counter.frames, "episode": episodes, "return": float(ep["r"]),
                       "length": int(ep["l"]), "time": round(time.time() - t0, 1),
                       **{k: float(np.mean(v)) for k, v in metrics.items()}}
                append_csv(os.path.join(run_dir, "train.csv"), row)
                extra = " ".join(f"{k}={v:.3g}" for k, v in row.items() if k not in ("frames", "episode", "return", "length", "time"))
                print(f"[{counter.frames:>9,}] ep {episodes:4d} return {row['return']:7.1f} "
                      f"({(counter.frames - start_frames) / max(row['time'], 1e-6):.0f} fps) {extra}")
                metrics.clear()
                obs, _ = env.reset(seed=train_track(seed, episodes))

            if counter.frames >= next_eval:
                next_eval += args.eval_every
                run_eval()
    except BudgetExceeded:
        pass

    if counter.frames > last_eval:  # 예산을 다 쓴 마지막 시점도 검증해서 best.pt 후보에 포함
        run_eval()
    save_checkpoint(latest_path, agent, seed, counter.frames, episodes, best_score)
    save_checkpoint(os.path.join(run_dir, "final.pt"), agent, seed, counter.frames, episodes, best_score)
    print(f"done: {counter.frames:,} frames, {episodes} episodes, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
