# HANDOFF — `05_wocar_dqn_r3` (rev3: prepared, not yet trained)

Read `README.md` (how to run), `docs/REV3.md` (every change vs rev2 and why). The earlier rev2 handoff text was replaced; rev2 itself lives untouched in `entries/04_wocar_dqn_r2/`.

## State
- Folder = copy of `04_wocar_dqn_r2` code (no artifacts, no notebook, no `smoke_test.py`) with: 3 changed configs, `run_orchestrator.py` / `monitoring.py` set to 3 seeds × 3 groups × 2M frames, new orchestrator flags (`--groups --seeds --concurrent`), and rev3 defaults in `evaluate.py` / `report.py`.
- Verified (this session): `run_orchestrator.py --dry-run` OK (also with `--groups/--seeds/--concurrent/--run-root` for the pilot); 4,000-frame smoke runs of all three configs finished and wrote `frame_4000.pth` + `train.log` (`artifacts/smoke/r3_*`), with the WocaR worst-case-loss path forced on by pulling the ramps forward (worst kappa 0.5, worst eps 0.005, worst-case loss finite);
  `evaluate.py` + `report.py` run end to end on those checkpoints (2-episode eval); `visualize_perturbation.py` runs against the rev3 env/config.
- **Not verified**: anything about learning (no real training has run), the Windows training machine's core count / speed with 25 vehicles, whether 25 vehicles is too dense (see pilot gate in README), whether κ = 0.02 / `worst_kappa_end` = 0.5 produce a visible gap.
- Committed on `wocar_r2` (pushed) and continued on branch `wocar_r3`. `artifacts/smoke/` is untracked scratch.
- `monitoring.py` now writes `PROGRESS.md`/`progress.json` into this entry folder (smoke: `artifacts/smoke/`) and uses its own lock file, so it cannot clobber rev2's repo-root files; the commit path was tested in a throwaway repo.
- Full pipeline smoke (`run_orchestrator.py --smoke`, 9 runs × 3,000 frames) passed on macOS: 9/9 finished, monitor shows 9/9 done.

## Decisions made (do not relitigate without the user)
- Reward definition unchanged. Lanes unchanged (y-normalisation depends on lane count, so ε keeps its physical meaning).
- 2,000,000 frames, save/mini-test interval 50,000 (final checkpoint = last mini-test). Episode length 60 is deliberately **not** aligned with frames (crashes end episodes early).
- 3 seeds per group. Evaluate the final frame of every run; no checkpoint selection (the peak-of-mean frame is shown for information only).
- Worst-case loss (`worst_kappa_end`) is on at 0.5 for WocaR: it is not one of the paper's three mechanisms but is used in upstream's published Atari configs (0.1–0.9); the README warns of instability. Guard: restart a diverging run with 0.9 or `worst_ratio` 0.

## Open items / next steps
1. Run the pilot described in the README (vanilla + SA-DQN, seed 1, 600k frames) and decide go / no-go for the full run; adjust `vehicles_count` if vanilla's clean crash rate is very high.
2. Launch the full run (`run_orchestrator.py`, optionally `--concurrent`; `monitoring.py` in a second window). Check the training gates in the README around 400k and 600k frames.
3. After training: `evaluate.py`, `report.py`, and compare vanilla vs SA-DQN vs WocaR at ε = 0.005 (training) and 0.01.
4. `worst_ratio` / `worst_eps` / `worst_kappa` ramp together from `schedule_start`; a delayed worst-critic ramp (as in the paper: off for the first third) needs a `train.py` change if wanted.
5. Possible issue to check in the paper: its Algorithm 2 puts κ_wst on the natural Q and (1−κ_wst) on the worst critic in the target mix, while the code mixes `(1−ratio)·natural + ratio·worst`; verify against the PDF before relying on the ratio semantics.

## Rules carried over from the user
- Never delete or move anything under `artifacts/`; commit only what is produced with explicit pathspecs (no `git add -A`); do not touch `PROGRESS.md` / `progress.json` (written by the monitor); do not add the large cumulative `.npy` files.
- Commit trailers used in this repo: `Co-Authored-By: <model name> <noreply@anthropic.com>`. The repo stores LF line endings.
- Report outcomes faithfully (failed gates included); ask before anything outward-facing (push, publish).
