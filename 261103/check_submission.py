"""공식 채점기 (ONNX 제출 파일용).  [수정 금지]

교수자 채점 서버도 이 파일을 그대로 사용합니다. 제출 전에 반드시 직접 돌려 보세요.
    python check_submission.py --onnx MyTeam.onnx
    python check_submission.py --onnx MyTeam.onnx --seeds 1000-1019

채점 방식:
    - CarRacing-v3 원본 환경 + HideScoreText(화면 아래 점수 숫자를 그리지 않음) (이 저장소의 envs.py는 사용하지 않음)
    - 최근 n_stack개 agent step의 원본 RGB 프레임 (1, n_stack, 96, 96, 3) uint8을 정책에 입력
      (에피소드 시작 시에는 첫 프레임을 n_stack번 반복)
    - 정책 출력 행동을 frame_skip 프레임 동안 반복
    - 점수 = 트랙(seed)별 원본 보상 return 평균

ONNX 파일 규격 (export.py가 자동으로 맞춰 줌):
    입력 "frames" (1, n_stack, 96, 96, 3) uint8
    출력 "action" 이산: (1,) int64 in [0, 4] / 연속: (1, 3) float32 (범위 밖 값은 잘라냄)
    metadata: continuous (true/false), frame_skip (1~8), n_stack (1~16)
"""

import argparse
import json
import os
import sys
from collections import deque

import gymnasium as gym
import numpy as np
import onnx
import onnxruntime as ort
import pygame

PUBLIC_EVAL_SEEDS = list(range(1000, 1010))
MAX_FILE_MB = 100


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


class SubmissionError(Exception):
    pass


def load_policy(path):
    if os.path.getsize(path) > MAX_FILE_MB * 1024 * 1024:
        raise SubmissionError(f"파일이 {MAX_FILE_MB}MB를 넘습니다. (File is larger than {MAX_FILE_MB}MB.)")
    try:
        model = onnx.load(path)
    except Exception as e:
        raise SubmissionError(f"ONNX 파일을 읽을 수 없습니다. (Cannot read the ONNX file.) {e}")
    meta = {p.key: p.value for p in model.metadata_props}
    try:
        cfg = {
            "continuous": meta["continuous"].lower() == "true",
            "frame_skip": int(meta["frame_skip"]),
            "n_stack": int(meta["n_stack"]),
        }
    except (KeyError, ValueError):
        raise SubmissionError("metadata(continuous, frame_skip, n_stack)가 없습니다. export.py로 만드세요. "
                              "(Missing metadata. Create the file with export.py.)")
    if not 1 <= cfg["frame_skip"] <= 8 or not 1 <= cfg["n_stack"] <= 16:
        raise SubmissionError(f"허용 범위를 벗어난 metadata (Metadata out of the allowed range): {cfg}")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    # 경로 대신 바이트로 전달: 일부 유니코드 파일 이름(예: macOS 한글)을 onnxruntime이 경로로 열지 못함
    sess = ort.InferenceSession(model.SerializeToString(), opts, providers=["CPUExecutionProvider"])
    inputs = [i.name for i in sess.get_inputs()]
    outputs = [o.name for o in sess.get_outputs()]
    if inputs != ["frames"] or "action" not in outputs:
        raise SubmissionError(f"입력은 'frames', 출력은 'action'이어야 합니다 (Input must be 'frames' and output 'action'): "
                              f"{inputs} -> {outputs}")
    return sess, cfg


def to_env_action(raw, cfg, env):
    a = np.asarray(raw).reshape(-1)
    if cfg["continuous"]:
        if a.shape != (3,) or not np.all(np.isfinite(a)):
            raise SubmissionError(f"연속 행동은 유한한 값 3개여야 합니다 (A continuous action must be 3 finite values): {a}")
        return np.clip(a.astype(np.float32), env.action_space.low, env.action_space.high)
    if a.shape != (1,) or not 0 <= int(a[0]) < env.action_space.n:
        raise SubmissionError(f"이산 행동은 0~{env.action_space.n - 1} 정수 하나여야 합니다 "
                              f"(A discrete action must be a single integer in 0~{env.action_space.n - 1}): {a}")
    return int(a[0])


def run_episode(sess, cfg, seed):
    env = HideScoreText(gym.make("CarRacing-v3", continuous=cfg["continuous"]))
    obs, _ = env.reset(seed=seed)
    frames = deque([obs] * cfg["n_stack"], maxlen=cfg["n_stack"])
    total, done = 0.0, False
    while not done:
        x = np.stack(frames)[None]  # (1, n_stack, 96, 96, 3) uint8
        action = to_env_action(sess.run(["action"], {"frames": x})[0], cfg, env)
        for _ in range(cfg["frame_skip"]):
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
            if done:
                break
        frames.append(obs)
    env.close()
    return total


def evaluate_onnx(path, seeds):
    sess, cfg = load_policy(path)
    return cfg, [run_episode(sess, cfg, s) for s in seeds]


def parse_seeds(s):
    if "-" in s:
        a, b = map(int, s.split("-"))
        return list(range(a, b + 1))
    return [int(x) for x in s.split(",")]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--seeds", type=parse_seeds, default=PUBLIC_EVAL_SEEDS)
    p.add_argument("--json", default=None, help="결과 JSON 저장 경로 (채점 서버용)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    try:
        cfg, returns = evaluate_onnx(args.onnx, args.seeds)
        result = {"ok": True, "config": cfg, "returns": returns,
                  "mean": float(np.mean(returns)), "std": float(np.std(returns))}
    except SubmissionError as e:
        result = {"ok": False, "error": str(e)}
    except Exception as e:  # 실행 중 오류 (shape 불일치 등)
        result = {"ok": False, "error": f"{type(e).__name__}: {e}"[:500]}

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
    if not args.quiet:
        if result["ok"]:
            print(f"설정: {result['config']}")
            if len(args.seeds) <= 20:
                for s, r in zip(args.seeds, result["returns"]):
                    print(f"  seed {s}: {r:8.1f}")
            print(f"SCORE: {result['mean']:.1f} ± {result['std']:.1f}  ({len(args.seeds)} tracks)")
        else:
            print(f"제출 파일 오류: {result['error']}")
    sys.exit(0 if result["ok"] else 1)
