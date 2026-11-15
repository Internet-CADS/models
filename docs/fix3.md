# Instruction for the training agent (2026-11-16): continue after gate 5 FAIL

The user has decided: **continue with stage 6 as planned, with the WocaR group that failed gate 5**
(`wocar_dqn_reg_b`, worst_ratio 0.2). You do not need to ask. Gate 5 stays recorded as FAIL; the override is recorded
separately as the user's decision.

Reason (for your RUNLOG entry, no action needed): WocaR is not broken. It is stable, certifies 81.6% and passed every
check except clean crash (25.3% vs the 21.8% limit). Its weaker clean driving is a result the paper needs with 6
seeds, and the stage 6 arm `wocar_sel_dqn` (docs/fix2.md) checks whether our target formula causes it.

**What to do** (from `entries/06_wocar_dqn_r4`):
1. Pull: `git -C ../.. pull --rebase --autostash origin rev4x` (you need the commit that adds this file).
2. Record the override (this sets `gate5_override` in `rev4_params.json` and writes a `RESUMED` entry to `RUNLOG.md`;
   do not edit either file by hand):
   ```
   python -c "import rev4; rev4.override_gate5('continue with stage 6 and wocar_dqn_reg_b despite the clean-crash FAIL (docs/fix3.md)')"
   ```
3. Commit and push:
   ```
   git -C ../.. add -- entries/06_wocar_dqn_r4/RUNLOG.md entries/06_wocar_dqn_r4/rev4_params.json
   git -C ../.. commit -m "RESUMED: user override of gate 5 (docs/fix3.md)"
   git -C ../.. push origin rev4x
   ```
4. Continue with **step 8** of `docs/rev4x_runbook.md` (stage 6), then steps 9 and 10, as written. The dry run
   (`python run_orchestrator.py --stage 6 --dry-run`) must list `sa_dqn_c030` seed 6, `wocar_dqn_reg_b` seeds 4–6 and
   `wocar_sel_dqn` seeds 1–2, with `training_config:worst_target_mode=select` only on the `wocar_sel_dqn` runs and
   `training_config:worst_ratio=0.2` on all WocaR runs. If it says `NOT ALLOWED ... gate5`, step 2 did not work: stop
   and ask.
5. In the final message (step 10), say that gate 5 failed twice on clean crash and that the user overrode it.

**If a pull or push gives a conflict in `RUNLOG.md`:** keep both sides' entries, in time order, and continue.
