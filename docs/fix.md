# Instruction for the training agent (2026-11-15, 06:00)

Do this now, then continue with the runbook as normal. You do not need to ask the user.

1. Pull: `git -C <repo> pull --rebase --autostash origin rev4x` (you need commit `254a8e2` or later).
2. Read the block "Correction of 2026-11-15, 06:00" in step 7 of `docs/rev4x_runbook.md` and follow it. In short:
   - The gate 5 evaluation picked WocaR checkpoints from before its training reached full strength. `evaluate.py`
     now only picks checkpoints after the training ramps.
   - If `evaluate.py --preset gate5` is still running, let it finish, but do not trust its WocaR numbers.
   - If `python gate.py stage5` has already run, undo only its decision:
     `git -C <repo> checkout e9f4a1e -- entries/06_wocar_dqn_r4/rev4_params.json`.
     If `wocar_dqn_reg_b` or stage 6 training has already started, stop and ask instead (procedure in the runbook).
   - From `entries/06_wocar_dqn_r4`: `python evaluate.py --preset gate5 --workers 6` (about 30 minutes; only the WocaR
     checkpoints at 700,000 frames are new), then `python gate.py stage5`, then continue with step 7 as written.
3. Record in `RUNLOG.md`: "gate 5 re-evaluated after the checkpoint correction (docs/fix.md)", and commit it with
   the gate 5 commit.
