# Instruction for the training agent (2026-11-15, 16:00): stage 6 exploratory arm changed

The user has approved this change. You do not need to ask.

**What changed:** stage 6's exploratory arm is now `wocar_sel_dqn` (seeds 1–2) instead of `wocar_wcl_dqn`. It is
WocaR-DQN with the target of the authors' code: the natural / worst-case mix only picks the next action, and the
natural Q value is bootstrapped (`worst_target_mode=select`). Everything else is the same as the main WocaR group,
including `worst_ratio` from `rev4_params.json`. It tells us whether our paper-style target is what costs WocaR its
clean driving. The code (`rev4.py`) handles it; the commands are unchanged.

**What to do:**
1. Now: pull (`git -C <repo> pull --rebase --autostash origin rev4x`). Do not touch the running stage 5 retry.
2. Read the new entry "Change to stage 6 (user, ...)" at the end of `entries/06_wocar_dqn_r4/RUNLOG.md`, then add an
   entry below it: "docs/fix2.md read; stage 6 will train wocar_sel_dqn seeds 1-2 instead of wocar_wcl_dqn".
   Commit (`log(rev4): fix2.md read`) and push.
3. Continue with the runbook as written (step 7: gate 5 after the retry).
4. Before launching stage 6 (step 8), pull again. The dry run (`python run_orchestrator.py --stage 6 --dry-run`) must
   list `sa_dqn_c030` seed 6, the WocaR group seeds 4–6 and `wocar_sel_dqn` seeds 1–2, with
   `training_config:worst_target_mode=select` only on the `wocar_sel_dqn` runs. If it lists `wocar_wcl_dqn`, you are
   running old code: pull and check again.
5. If gate 5 FAILS (BLOCKED), stop and ask as the runbook says; do not start `wocar_sel_dqn` on your own.

**If a pull or push gives a conflict in `RUNLOG.md`:** keep both sides' entries, in time order, and continue.
