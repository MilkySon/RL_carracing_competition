# A2C Legacy → GAE → PPO → Coupled Action (CarRacing-v3)

Professor baseline is retained under `baseline/`. Protected wrappers and `check_submission.py` are **unchanged**. Only new `baseline/agents/legacy_extended.py` and new orchestration scripts are added.

## Colab notebooks
Open `notebooks/E0_A2C_Colab.ipynb`, `E1_GAE_Colab.ipynb`, `E2_PPO_Colab.ipynb`, or `E3_ActionSpace_Colab.ipynb`. Set `REPO_URL` to the GitHub repository **after committing this folder**. Each notebook clones the repository and invokes the runner.

## Experiments
- E0: professor original A2C, continuous=True
- E1: same A2C + GAE(0.95)
- E2: E1 + PPO clipped objective, 4 epochs, minibatch 64, target KL 0.02
- E3: E2 + 2D steering/signed throttle with fixed neutral band 0.1

Training and evaluation use the original baseline's `train.py`/`evaluate.py`. Each experiment uses the same 300,000 raw-frame budget (evaluation frames excluded). Validation uses seeds 500–504; diagnostic/public videos use 1000–1009. Best/worst means **lowest/highest raw return among those ten public tracks**, not training performance.

## Outputs
`outputs/E2/seed_0/`: `train.csv`, `eval.csv`, `best.pt`, `latest.pt`, `final.pt`, `reports/eval_episodes.csv`, `reports/eval_summary.json`, `reports/eval_events.csv`, `reports/action_trace.csv`, `reports/check_submission.json`, `videos/worst_seed_*/`, `videos/best_seed_*/`.

`outputs/submissions/Reinforcement_material_E2_seed0.onnx` (auto version suffix `_v2` etc if a name exists). File is published only **after** `export.py` and original `check_submission.py` both succeed. Keep the original `.onnx` for submission; do not rename internal input/output tensors.

## Diagnostic caveats
- `mean_return`: unscaled `info['episode']['r']`.
- `completion_rate`: `tile_visited_count >= len(track)` as completion proxy; check environment behavior on a sample video.
- `offroad_recovery_rate`: closest-track-center distance proxy, return to road / all observed offroad entries; unfinished exits count as non-recovery.
- `gas_brake_overlap_rate`: fraction of raw environment frames where both gas/brake >0.1 (actual frame counts used).
- `curve_failure_rate`: offroad entries occurring in high-curvature proxy / detected curve entries; proxy threshold 0.16 radians over ±3 track nodes. This is **not** an official metric and should be validated manually.
- `sample_efficiency`: `eval.csv` gives validation mean return by raw frame count. AUC can be computed from it.
- Diagnostics inspect environment internals **only after action selection**. No diagnostic data is fed to the agent, reward, or action. Never add hand-coded steering, curve, or grass recovery overrides.

## Colab dependencies and storage
Colab installs baseline requirements; outputs saved in the clone's `outputs/` directory. For persistent outputs, mount Google Drive and set `OUTPUT_ROOT` in the notebook. `--resume` uses latest.pt; it is **not guaranteed bitwise deterministic** across interrupted sessions because baseline does not serialize the live environment and RNG state. A new run directory is safer for strict reproducibility.

## Quick local usage
```bash
cd experiments/continuous_ablation
python scripts/run_experiment.py --experiment E1 --team Reinforcement_material --seed 0 --budget 300000
```

## Rule compliance
Original `baseline/check_submission.py` and protected blocks in `baseline/envs.py` unchanged. Learned policy determines every action. E3 uses a fixed global policy parameterization, never conditional steering or recovery overrides. Confirm action parameterization with the instructor if interpretation of rule 1 is uncertain.
