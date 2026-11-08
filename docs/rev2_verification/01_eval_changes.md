# 01 — mini_test safety-score removal (Gate 1)

## Where the score logic lived
| File | Location (pre-change line) | What it was |
|---|---|---|
| `train.py` `main()` | 882 | `best_test_reward = -float('inf')` running-best state |
| `train.py` `main()` | 1136–1137 | `safety_score = test_reward - 5.0 * crash_rate`; `max_crash_rate` from `safety_max_crash_rate` |
| `train.py` `main()` | 1138–1139 | train.log row carried `safety score: …, eligible: …` |
| `train.py` `main()` | 1143 | TB scalar `test/safety_score` |
| `train.py` `main()` | 1145–1151 | `if safety_score >= best_test_reward:` block: "new best safety score" log line, `torch.save(... 'best_frame_<N>.pth')`, TB scalar `test/best_safety_score` |
| `config_highway_{vanilla,sadqn,wocar}.json` | `training_config.safety_max_crash_rate` (0.10) | only consumer was the deleted `eligible` flag |

## Deleted (no stubs)
All of the above. Net diff: `train.py` −13/+1 lines; each config −1 key (trailing comma on the preceding key removed, JSON re-validated). No other hyperparameter touched.

## Kept unchanged
- `mini_test()` and its call (every `mini_test: 50000` frames, 50 episodes) — still returns `(reward, crash_rate)`.
- train.log row is now `this test avg reward: <r>, crash rate: <c>` — the `this test avg reward:` prefix is intact for parsers.
- TB scalars `test/return`, `test/crash_rate`, `test/length` kept.
- **50k cadence unchanged:** `frame_<N>.pth` save (`train.py` l.1117–1120, `save_frame` gate) and `mini_test` interval untouched.
- `upstream/` untouched.

## Verification
- `grep -i "safety|best_"` over `train.py` and the three configs → no matches.
- `py_compile train.py` OK. (No training run yet; that is Gate 3.)

## TRAIN_SEED plumbing
The existing CLI (`upstream/argparser.py`, vendored, not modified) already accepts `--path_prefix <dir>` and positional `key:subkey=value` overrides, applied by `read_config.load_config`. So per-run seed and `num_frames` need no train.py change:
`python train.py --config config_highway_<m>.json --path_prefix <dir> training_config:seed=<1..6>`
`training_config.seed` seeds `env.seed`, `np`, `random`, `torch` (`train.py` ~l.685–689); the full "total seeding" audit is Gate 2.
Note: `train.py` writes to `<path_prefix>/highway-v0_<name_suffix>/`, so the per-run dir is `path_prefix` plus that suffix folder — to be resolved when the orchestrator layout is built (Gate 4). The `path_prefix` in the configs is **not** repointed yet.

## Not changed in this gate (flagged)
- `test.py` l.97–100 (evaluation script) still prefers `best_frame_*.pth` when present. Eval is out of scope; no such files will exist after this change.
- `docs/archive/train-highway-3rd-run-cpu.md` (legacy runbook) still describes best-frame/selection; it is discarded per the brief.
- `.gitignore` `*.log` removal and `path_prefix` repoint belong to later gates.
