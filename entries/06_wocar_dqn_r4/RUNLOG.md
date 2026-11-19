# Rev 4 run log

Progress file of the rev 4 training (`docs/rev4x_runbook.md`). Newest entries are at the bottom.
The orchestrator, `evaluate.py` and `gate.py` append here by themselves; the training agent adds a line when it
starts a step or sees a problem: `python -c "import rev4; rev4.log('...')"`.
Live training progress is in `PROGRESS.md`; the gates' decisions are in `rev4_params.json`.

### 2026-11-14 00:37
Step 0 done: imports, dry run and smoke run ok

### 2026-11-14 00:37
Step 1 started: stage 1 training launched

### 2026-11-14 00:37
Step 1c: CPU affinity 0x555 applied to train.py processes (6 cores/12 threads confirmed)

### 2026-11-14 00:48
Step 1c check: ~68 f/s per run with affinity 0x555 (expected 55-65), no change needed

### 2026-11-14 03:43
**Interim mini-test results, stage 1 (informational, not a gate)** - 14 tests per run, i.e. up to 700,000 frames (mini-test every 50,000 frames, 50 episodes each). Cells are return / crash rate.

| run | 50k | 100k | 150k | 200k | 250k | 300k | 350k | 400k | 450k | 500k | 550k | 600k | 650k | 700k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vanilla_g097/model_1 | 9.3 / 0.10 | 9.3 / 0.72 | 12.5 / 0.32 | 16.2 / 0.26 | 19.1 / 0.34 | 21.8 / 0.24 | 22.5 / 0.22 | 23.1 / 0.22 | 25.0 / 0.14 | 25.0 / 0.08 | 24.4 / 0.16 | 24.9 / 0.16 | 25.0 / 0.12 | 26.4 / 0.10 |
| vanilla_g097/model_2 | 8.5 / 0.10 | 10.7 / 0.54 | 14.6 / 0.24 | 16.8 / 0.42 | 20.9 / 0.24 | 18.4 / 0.42 | 18.2 / 0.36 | 21.2 / 0.30 | 22.5 / 0.22 | 22.7 / 0.26 | 24.9 / 0.16 | 22.3 / 0.26 | 24.8 / 0.10 | 26.2 / 0.10 |
| vanilla_g097/model_3 | 6.8 / 0.08 | 10.4 / 0.50 | 10.9 / 0.70 | 16.6 / 0.38 | 19.4 / 0.30 | 19.0 / 0.34 | 20.2 / 0.30 | 21.6 / 0.28 | 22.9 / 0.18 | 25.1 / 0.12 | 22.3 / 0.26 | 25.9 / 0.06 | 23.6 / 0.20 | 25.2 / 0.14 |
| vanilla_g090/model_1 | 11.3 / 0.58 | 14.4 / 0.62 | 21.2 / 0.30 | 22.8 / 0.24 | 24.8 / 0.16 | 23.5 / 0.22 | 23.9 / 0.18 | 23.8 / 0.18 | 26.6 / 0.08 | 25.4 / 0.10 | 24.9 / 0.12 | 25.2 / 0.14 | 26.9 / 0.08 | 25.7 / 0.12 |
| vanilla_g090/model_2 | 12.8 / 0.24 | 16.8 / 0.44 | 19.4 / 0.34 | 22.1 / 0.26 | 23.8 / 0.16 | 23.8 / 0.18 | 24.7 / 0.14 | 22.1 / 0.26 | 24.0 / 0.20 | 24.0 / 0.16 | 25.3 / 0.12 | 25.7 / 0.12 | 26.3 / 0.12 | 24.1 / 0.16 |
| vanilla_g090/model_3 | 12.1 / 0.48 | 15.1 / 0.58 | 19.2 / 0.32 | 22.1 / 0.26 | 23.7 / 0.20 | 24.0 / 0.16 | 24.6 / 0.16 | 22.7 / 0.24 | 25.9 / 0.10 | 26.0 / 0.12 | 26.4 / 0.10 | 27.2 / 0.04 | 25.1 / 0.16 | 26.9 / 0.08 |

Gate-1-style tail (mean of last 3 mini-tests per seed, then mean over seeds). Gate 1 needs crash <= 15%, return >= 18, and Q/return in [0.6, 1.5] (Q/return only known after diagnose.py):

| arm | return | crash | per-seed crash |
|---|---|---|---|
| vanilla_g097 | 24.9 | 13.8% | 12.7% / 15.3% / 13.3% |
| vanilla_g090 | 25.9 | 11.3% | 11.3% / 13.3% / 9.3% |

Observations: all six runs improve steadily, no late collapse seen; gamma 0.90 learns faster early, gamma 0.97 catches up by ~600-700k. Single readings are noisy (+-0.1-0.2 crash rate). Agent's expectation (not a decision): gate 1 likely PASS with vanilla_g090, main uncertainty is the Q/return check and tail noise.

### 2026-11-14 03:58
Pulled b1a9b26 (finer epsilon grids) with pull --rebase --autostash, fast-forward, nothing restarted. EPSILON_SEARCH now [0.0, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.05, 0.1]; eval epsilons now 0 + (0.25,0.5,0.75,1,1.5,2) x epsilon. Training at ~77%, 6 runs alive.

### 2026-11-14 04:51
Training stage 1: 6/6 runs finished
- vanilla_g097/model_1: finished 
- vanilla_g097/model_2: finished 
- vanilla_g097/model_3: finished 
- vanilla_g090/model_1: finished 
- vanilla_g090/model_2: finished 
- vanilla_g090/model_3: finished

### 2026-11-14 05:10
Step 1e done: stage 1 finished (6/6 runs at 1,000,000 frames, no failed/stalled runs), monitor stopped. Next: stage 1 data commit, then gate 1 (diagnose.py --stage1, gate.py stage1).

### 2026-11-14 05:11
**Gate `stage1`: PASS**

| run | gamma | crash (last 3 mini-tests) | return | Q / return | Q before crash | mean Q | crash sensitivity | median action gap |
|---|---|---|---|---|---|---|---|---|
| vanilla_g097/model_1 | 0.97 | 15.3% | 25.2 | 1.05 | 10.36 | 17.10 | 0.35 | 0.043 |
| vanilla_g097/model_2 | 0.97 | 17.3% | 24.5 | 1.02 | 2.77 | 16.75 | 0.75 | 0.070 |
| vanilla_g097/model_3 | 0.97 | 8.0% | 26.0 | 1.06 | 10.24 | 16.97 | 0.35 | 0.063 |
| vanilla_g090/model_1 | 0.9 | 10.0% | 26.4 | 1.03 | 1.96 | 4.87 | 0.42 | 0.029 |
| vanilla_g090/model_2 | 0.9 | 20.0% | 24.2 | 1.03 | 2.88 | 4.86 | 0.29 | 0.032 |
| vanilla_g090/model_3 | 0.9 | 12.7% | 26.0 | 1.05 | 2.08 | 4.87 | 0.41 | 0.033 |

| arm | mean crash | mean return | mean Q / return | passes |
|---|---|---|---|---|
| vanilla_g097 | 13.6% | 25.2 | 1.05 | yes |
| vanilla_g090 | 14.2% | 25.5 | 1.04 | yes |

Rule: crash <= 15.0%, return >= 18, 0.6 <= Q/return <= 1.5; among passing arms the lowest crash rate wins (ties within 2 points: the higher return).
Crash sensitivity = (mean Q - Q one step before a crash) / (mean Q + 2): the share of a crash's cost the value function shows just before it (information only; rev 3 vanilla: 0.20, laptop pilot: 0.30-0.45).

Chosen arm: **vanilla_g090** (gamma 0.9). hinge_c for the robust models: **0.031** (mean over seeds of the median action gap: 0.031, clipped to (0.02, 0.5)).

Next: `python evaluate.py --preset epsilon --workers 6`, then `python gate.py epsilon`.

### 2026-11-14 05:11
Step 3 started: epsilon sweep (evaluate.py --preset epsilon, then gate.py epsilon)

### 2026-11-14 05:33
Evaluation preset `epsilon`: 3 jobs done in 22 min (100 episodes, epsilons [0.0, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.05, 0.1]).

### 2026-11-14 05:33
**Gate `epsilon`: PASS**

Vanilla arm vanilla_g090, 3 seeds, 100 episodes each.

| epsilon | x / y / speed error | crash, no attack | crash, gradient attacks | attack-induced | random-noise-induced | flipped (PGD) | usable |
|---|---|---|---|---|---|---|---|
| 0.005 | 1 m / 0.08 m / 0.4 m/s | 15.3% | 52.9% | 37.6% | 1.7% | 56.4% | yes |
| 0.0075 | 1.5 m / 0.12 m / 0.6 m/s | 15.3% | 74.8% | 59.4% | 4.7% | 65.7% | yes |
| 0.01 | 2 m / 0.16 m / 0.8 m/s | 15.3% | 85.6% | 70.2% | 7.7% | 72.1% | yes |
| 0.015 | 3 m / 0.24 m / 1.2 m/s | 15.3% | 96.2% | 80.9% | 13.7% | 82.3% | yes |
| 0.02 | 4 m / 0.32 m / 1.6 m/s | 15.3% | 97.7% | 82.3% | 23.0% | 86.1% | yes |
| 0.05 | 10 m / 0.8 m / 4 m/s | 15.3% | 95.9% | 80.6% | 62.0% | 96.7% | yes |
| 0.1 | 20 m / 1.6 m / 8 m/s | 15.3% | 81.0% | 65.7% | 82.3% | 99.8% | no |

Rule: the smallest epsilon <= 0.05 at which gradient attacks add at least 20.0% to vanilla's crash rate.

Chosen epsilon: **0.005** (robust training and evaluation at [0.00125, 0.0025, 0.00375, 0.005, 0.0075, 0.01]).

Next: `python run_orchestrator.py --stage 2` (runbook step 4).

### 2026-11-14 05:33
Step 4 started: stage 2 training (orchestrator + monitor as managed background shells, affinity 0x555 after launch)

### 2026-11-14 05:34
Step 4 / 1c: stage 2 launched at 05:33 (vanilla_g090 seeds 4-6, sa_dqn seeds 1-3), affinity 0x555 applied. Gate epsilon: PASS, epsilon=0.005.

### 2026-11-14 10:42
Training stage 2: 6/6 runs finished
- vanilla_g090/model_4: finished 
- vanilla_g090/model_5: finished 
- vanilla_g090/model_6: finished 
- sa_dqn/model_1: finished 
- sa_dqn/model_2: finished 
- sa_dqn/model_3: finished

### 2026-11-14 10:42
Step 4: stage 2 training finished 10:42 (6/6 runs at 1,000,000 frames: vanilla_g090 seeds 4-6, sa_dqn seeds 1-3, none failed), monitor stopped. Next: stage 2 data commit, evaluate.py --preset gate2, gate.py stage2.

### 2026-11-14 10:43
Step 4: stage 2 data committed (8ad266d). Starting evaluate.py --preset gate2 (managed background shell).

### 2026-11-14 11:40
Evaluation preset `gate2`: 9 jobs done in 57 min (100 episodes, epsilons [0.0, 0.00125, 0.0025, 0.00375, 0.005, 0.0075, 0.01]).

### 2026-11-14 11:40
**Gate `stage2`: PASS**

At the training epsilon 0.005 (100 episodes per seed, peak-of-mean checkpoints).

| group | seeds | crash, no attack | return, no attack | crash, gradient attacks | attack-induced | flipped (PGD) | certified |
|---|---|---|---|---|---|---|---|
| Vanilla DQN (gamma 0.9) | 6 | 13.8% | 25.2 | 48.6% | 34.8% | 55.8% | 0.0% |
| SA-DQN | 3 | 13.0% | 25.7 | 45.1% | 32.1% | 9.9% | 65.9% |

| check | rule | result |
|---|---|---|
| clean crash | <= vanilla + 8.0% | ok |
| clean return | >= 0.85 x vanilla | ok |
| certified | >= 50.0% | ok |
| flip rate | <= 0.5 x vanilla | ok |

kappa = 0.02, hinge_c = 0.031, group folder = sa_dqn. The crash numbers under attack are shown for information; they are not part of this gate (3 seeds are too few).

Next: `python run_orchestrator.py --stage 3` (runbook step 5).

### 2026-11-14 11:40
Step 5 started: stage 3 training (wocar_dqn + more sa_dqn seeds) after gate 2 PASS. Note for final report: SA-DQN crash under gradient attack 45.1% vs vanilla 48.6% despite 65.9% certified and flip rate 9.9% vs 55.8%.

### 2026-11-14 11:40
Step 5 / 1c: stage 3 launched, affinity 0x555 applied to train.py processes.

### 2026-11-14 15:39
Runbook change read and understood (commit 4c1c90b, docs/rev4x_regulariser_plan.md). Stage 3 (sa_dqn 4-6 + wocar_dqn 1-3) continues unchanged, ~68% done, 56-59 f/s; its gate 3 is now informational (PASS/NOTED, always go on to step 6, no retry). New plan: stage 4 = sa_dqn_c030 / sa_dqn_k010 / sa_dqn_c030_k010 x seeds 1-2 then evaluate --preset reg and gate reg; stage 5 = chosen SA arm seeds 3-5 + wocar_dqn_reg 1-3 then gate5 evaluation and gate stage5; stage 6 = chosen SA seed 6 + wocar 4-6 + wocar_wcl 1-2; final evaluation + ablation + report. Before the gate 3 evaluation I will pull again.

### 2026-11-14 17:08
Training stage 3: 6/6 runs finished
- sa_dqn/model_4: finished 
- sa_dqn/model_5: finished 
- sa_dqn/model_6: finished 
- wocar_dqn/model_1: finished 
- wocar_dqn/model_2: finished 
- wocar_dqn/model_3: finished

### 2026-11-14 17:09
Step 5: stage 3 training finished 17:08 (6/6 runs at 1,000,000 frames: sa_dqn seeds 4-6, wocar_dqn seeds 1-3, none failed), monitor stopped. Next: stage 3 data commit, pull, evaluate --preset gate3, gate stage3 (informational).

### 2026-11-14 17:09
Step 5: stage 3 data committed (c367858), pulled (up to date). Starting evaluate.py --preset gate3.

### 2026-11-14 18:04
Evaluation preset `gate3`: 9 jobs done in 55 min (100 episodes, epsilons [0.0, 0.00125, 0.0025, 0.00375, 0.005, 0.0075, 0.01]).

### 2026-11-14 18:04
**Gate `stage3`: NOTED**

At the training epsilon 0.005 (100 episodes per seed, peak-of-mean checkpoints).

| group | seeds | crash, no attack | return, no attack | crash, gradient attacks | attack-induced | flipped (PGD) | certified |
|---|---|---|---|---|---|---|---|
| Vanilla DQN (gamma 0.9) | 6 | 13.8% | 25.2 | 48.6% | 34.8% | 55.8% | 0.0% |
| SA-DQN, first setting | 6 | 17.2% | 24.8 | 43.5% | 26.3% | 9.4% | 65.8% |
| WocaR-DQN, first setting | 3 | 23.7% | 23.5 | 47.3% | 23.7% | 20.6% | 6.0% |

| check | rule | result |
|---|---|---|
| clean crash | <= vanilla + 8.0% | FAILED |
| clean return | >= 0.85 x vanilla | ok |
| certified | >= 50.0% | FAILED |
| worst-case critic | last worst-Q of every seed in [-5, 6]: -0.17, 0.08, 0.17 | ok |

worst_ratio = 0.5, hinge_c = 0.031, kappa = 0.02, group folder = wocar_dqn. This gate is informational: these groups are the first regulariser setting, kept for the ablation. The run continues with stage 4 whatever the checks say (docs/rev4x_regulariser_plan.md).

Next: THE RUNBOOK CHANGED on 2026-11-14: read docs/rev4x_runbook.md again from step 6, then `python run_orchestrator.py --stage 4` (runbook step 6: SA-DQN with a stronger regulariser).

### 2026-11-14 18:04
Step 5 done: gate 3 NOTED (informational; first-setting SA-DQN 6 seeds certified 65.8%, WocaR first setting 3 seeds certified 6.0%, clean crash 23.7%). Step 6 started: stage 4 (sa_dqn_c030, sa_dqn_k010, sa_dqn_c030_k010 x seeds 1-2), dry run listed exactly these 6 runs.

### 2026-11-14 18:05
Step 6 / 1c: stage 4 launched, affinity 0x555 applied to train.py processes.

### 2026-11-14 23:17
Training stage 4: 6/6 runs finished
- sa_dqn_c030/model_1: finished 
- sa_dqn_c030/model_2: finished 
- sa_dqn_k010/model_1: finished 
- sa_dqn_k010/model_2: finished 
- sa_dqn_c030_k010/model_1: finished 
- sa_dqn_c030_k010/model_2: finished

### 2026-11-14 23:17
Step 6: stage 4 training finished 23:17 (6/6 runs at 1,000,000 frames: sa_dqn_c030, sa_dqn_k010, sa_dqn_c030_k010 x seeds 1-2, none failed), monitor stopped. Next: stage 4 data commit, pull, evaluate --preset reg, gate reg.

### 2026-11-14 23:17
Step 6: stage 4 data committed (4044d4e), pulled (up to date). Starting evaluate.py --preset reg.

### 2026-11-14 23:48
Evaluation preset `reg`: 6 jobs done in 31 min (100 episodes, epsilons [0.0, 0.00125, 0.0025, 0.00375, 0.005, 0.0075, 0.01]).

### 2026-11-14 23:48
**Gate `reg`: PASS**

At the training epsilon 0.005 (100 episodes per seed, peak-of-mean checkpoints). First setting (sa_dqn): hinge_c 0.031, kappa 0.02.

| group | seeds | crash, no attack | return, no attack | crash, gradient attacks | attack-induced | flipped (PGD) | certified |
|---|---|---|---|---|---|---|---|
| Vanilla DQN (gamma 0.9) | 6 | 13.8% | 25.2 | 48.6% | 34.8% | 55.8% | 0.0% |
| SA-DQN, first setting | 6 | 17.2% | 24.8 | 43.5% | 26.3% | 9.4% | 65.8% |
| SA-DQN, hinge_c 0.3 | 2 | 14.0% | 25.2 | 42.5% | 28.5% | 9.8% | 74.2% |
| SA-DQN, kappa 0.1 | 2 | 26.0% | 22.6 | 43.0% | 17.0% | 6.5% | 83.3% |
| SA-DQN, hinge_c 0.3 + kappa 0.1 | 2 | 30.0% | 21.0 | 49.3% | 19.3% | 6.3% | 86.6% |

| arm | hinge_c | kappa | clean crash <= vanilla + 8.0% | clean return >= 0.85 x vanilla | certified |
|---|---|---|---|---|---|
| sa_dqn_c030 | 0.3 | 0.02 | ok | ok | 74.2% |
| sa_dqn_k010 | 0.031 | 0.1 | FAILED | ok | 83.3% |
| sa_dqn_c030_k010 | 0.3 | 0.1 | FAILED | FAILED | 86.6% |

Rule: among the arms that drive about as well as vanilla, the highest certified share wins (within 2.0%: the lower clean crash rate); it must be at least 5.0% above the first setting. Target: 90.0%. The crash numbers under attack are shown for information; they are deliberately not part of the rule.

Chosen arm: **sa_dqn_c030** (hinge_c **0.3**, kappa **0.02**, certified 74.2%). It is now the SA-DQN group of the comparison; WocaR-DQN is trained with the same regulariser into `wocar_dqn_reg`.
Note: the chosen arm is below the 90.0% target. The run continues; mention this in the final message.

Next: `python run_orchestrator.py --stage 5` (runbook step 7).

### 2026-11-14 23:49
Step 6 done: gate reg PASS, chose sa_dqn_c030 (hinge_c 0.3, kappa 0.02, certified 74.2%, below the 90% target; first setting 65.8%). Note to repeat in final message: chosen arm below 90% certified target. Rejected: sa_dqn_k010 (clean crash 26.0%) and sa_dqn_c030_k010 (clean crash 30.0%, return 21.0) failed the driving checks. Step 7 started: stage 5 (sa_dqn_c030 seeds 3-5 + wocar_dqn_reg seeds 1-3).

### 2026-11-14 23:49
Step 7 / 1c: stage 5 launched, affinity 0x555 applied to train.py processes.

### 2026-11-15 05:13
Training stage 5: 6/6 runs finished
- sa_dqn_c030/model_3: finished 
- sa_dqn_c030/model_4: finished 
- sa_dqn_c030/model_5: finished 
- wocar_dqn_reg/model_1: finished 
- wocar_dqn_reg/model_2: finished 
- wocar_dqn_reg/model_3: finished

### 2026-11-15 05:14
Step 7: stage 5 training finished 05:13 (6/6 runs at 1,000,000 frames: sa_dqn_c030 seeds 3-5, wocar_dqn_reg seeds 1-3, none failed), monitor stopped. Next: stage 5 data commit, pull, evaluate --preset gate5, gate stage5.

### 2026-11-15 05:14
Step 7: stage 5 data committed (e9f4a1e), pulled (up to date). Starting evaluate.py --preset gate5.

### 2026-11-15 05:41
docs/fix.md read (pulled 254a8e2/ba8cd56). Gate 5 evaluation started with old code is still running (job 1/8, WocaR at 300000 = pre-ramp checkpoint) and is being left to finish; its WocaR numbers are not trusted. gate.py stage5 has NOT been run, wocar_dqn_reg_b and stage 6 do not exist, so no decision to undo. Next: when it finishes, rerun evaluate.py --preset gate5 with the corrected code (WocaR at 700000), then gate.py stage5.

### 2026-11-15 06:11
Evaluation preset `gate5`: 8 jobs done in 57 min (100 episodes, epsilons [0.0, 0.00125, 0.0025, 0.00375, 0.005, 0.0075, 0.01]).

### 2026-11-15 06:11
Old-code gate 5 evaluation finished (WocaR at 300000, discarded). Rerunning evaluate.py --preset gate5 with corrected code (254a8e2).

### 2026-11-15 06:36
Evaluation preset `gate5`: 3 jobs done in 24 min (100 episodes, epsilons [0.0, 0.00125, 0.0025, 0.00375, 0.005, 0.0075, 0.01]).

### 2026-11-15 06:36
**Gate `stage5`: RETRY**

At the training epsilon 0.005 (100 episodes per seed, peak-of-mean checkpoints).

| group | seeds | crash, no attack | return, no attack | crash, gradient attacks | attack-induced | flipped (PGD) | certified |
|---|---|---|---|---|---|---|---|
| Vanilla DQN (gamma 0.9) | 6 | 13.8% | 25.2 | 48.6% | 34.8% | 55.8% | 0.0% |
| SA-DQN, hinge_c 0.3 | 5 | 16.6% | 25.0 | 42.7% | 26.1% | 9.5% | 75.2% |
| WocaR-DQN | 3 | 32.7% | 21.7 | 62.6% | 29.9% | 8.1% | 80.2% |

| check | rule | result |
|---|---|---|
| clean crash | <= vanilla + 8.0% | FAILED |
| clean return | >= 0.85 x vanilla | ok |
| flip rate | <= 0.5 x vanilla | ok |
| attack-induced crashes | <= vanilla | ok |
| worst-case critic | last worst-Q of every seed in [-5, 6]: 0.19, -1.37, 0.25 | ok |

worst_ratio = 0.5, hinge_c = 0.3, kappa = 0.02, group folder = wocar_dqn_reg. The certified share is shown for information: WocaR's state-importance weighting leaves states with near-tied actions uncertified by design. Whether WocaR beats SA-DQN is the result of the experiment, not a gate.

One retry with a weaker worst-case mixing (leans less on the worst-case value, mainly for clean driving): worst_ratio 0.5 -> **0.2**, new group folder `wocar_dqn_reg_b`.

Next: repeat runbook step 7 (`python run_orchestrator.py --stage 5` now trains `wocar_dqn_reg_b` seeds 1-3; the SA-DQN runs are skipped).

### 2026-11-15 06:36
gate 5 re-evaluated after the checkpoint correction (docs/fix.md)

### 2026-11-15 06:36
Step 7 retry: gate 5 RETRY (WocaR clean crash 32.7% > 21.8% limit; flip rate, attack-induced crashes and worst-case critic ok; certified 80.2% informational). worst_ratio 0.5 -> 0.2, new group wocar_dqn_reg_b. Repeating stage 5: only wocar_dqn_reg_b seeds 1-3 train, finished sa_dqn_c030 runs are skipped by the orchestrator.

### 2026-11-15 06:36
Step 7 retry launched 06:36: wocar_dqn_reg_b seeds 1-3 (3 runs; sa_dqn_c030 3-5 skipped, logs kept). Affinity 0x555 applied to train.py processes.

### 2026-11-15 14:05
RESTARTED 2026-11-15 14:05: the desktop was force-restarted at ~07:17 and killed the 3 wocar_dqn_reg_b runs (logs only past ~100k frames, no failed-run tracebacks, no gate affected). train.py cannot resume, and evaluate/gate/report do not read retry_* dirs, so with the user permission to clean up the 3 partial attempts were MOVED (not deleted, untracked, 6.7 MB) to artifacts/aborted_after_restart_2026-11-15/wocar_dqn_reg_b/. Relaunching stage 5 (wocar_dqn_reg_b seeds 1-3 from scratch, worst_ratio 0.2). Everything else on rev4x intact: all commits pushed, gates 1-3, eps, reg, 5 (RETRY) unchanged.

### 2026-11-15 14:06
Stage 5 (wocar_dqn_reg_b seeds 1-3) relaunched after restart, affinity 0x555 applied.

### 2026-11-15 16:00
Change to stage 6 (user, made by Claude on the laptop; instruction in docs/fix2.md): the exploratory arm is now `wocar_sel_dqn` seeds 1-2 instead of `wocar_wcl_dqn`. It is WocaR-DQN with the authors' code's target (`worst_target_mode=select`: the natural / worst-case mix only picks the next action, the natural Q value is bootstrapped); everything else, including worst_ratio, is the same as the main WocaR group. Reason: WocaR with our paper-style target (`value`) crashes 32.7% without attack at gate 5 (vanilla 13.8%); this arm tells whether the target formula is the cause. Code: rev4.py (EXPLORATORY_GROUP, overrides); 3k-frame smoke run OK on the laptop. The running stage 5 retry and gate 5 are unaffected.

### 2026-11-15 19:12
Training stage 5: 6/6 runs finished
- sa_dqn_c030/model_3: finished already complete on disk
- sa_dqn_c030/model_4: finished already complete on disk
- sa_dqn_c030/model_5: finished already complete on disk
- wocar_dqn_reg_b/model_1: finished 
- wocar_dqn_reg_b/model_2: finished 
- wocar_dqn_reg_b/model_3: finished

### 2026-11-15 19:13
Step 7 retry: stage 5 retry finished 19:12 (wocar_dqn_reg_b seeds 1-3 at 1,000,000 frames, none failed; sa_dqn_c030 3-5 already complete), monitor stopped. Next: stage 5 data commit, pull, evaluate --preset gate5, gate stage5 (second attempt: a second failure is BLOCKED).

### 2026-11-15 19:13
Step 7 retry: data committed (7598c09), pulled (up to date). Starting evaluate.py --preset gate5 for wocar_dqn_reg_b.

### 2026-11-15 19:34
docs/fix2.md read; stage 6 will train wocar_sel_dqn seeds 1-2 instead of wocar_wcl_dqn. Code already present locally (rev4.py EXPLORATORY_GROUP=wocar_sel_dqn, worst_target_mode=select only for that group; commit 6f0fa73). Stage 5 retry data is committed (7598c09) and the gate 5 evaluation for wocar_dqn_reg_b is running; untouched. Before launching stage 6: pull again and check that the dry run lists sa_dqn_c030 seed 6, wocar_dqn_reg_b seeds 4-6 and wocar_sel_dqn seeds 1-2 only. If gate 5 BLOCKS, stop and ask; do not start wocar_sel_dqn on my own.

### 2026-11-15 19:38
Evaluation preset `gate5`: 3 jobs done in 26 min (100 episodes, epsilons [0.0, 0.00125, 0.0025, 0.00375, 0.005, 0.0075, 0.01]).

### 2026-11-15 19:39
**Gate `stage5`: FAIL**

At the training epsilon 0.005 (100 episodes per seed, peak-of-mean checkpoints).

| group | seeds | crash, no attack | return, no attack | crash, gradient attacks | attack-induced | flipped (PGD) | certified |
|---|---|---|---|---|---|---|---|
| Vanilla DQN (gamma 0.9) | 6 | 13.8% | 25.2 | 48.6% | 34.8% | 55.8% | 0.0% |
| SA-DQN, hinge_c 0.3 | 5 | 16.6% | 25.0 | 42.7% | 26.1% | 9.5% | 75.2% |
| WocaR-DQN (worst_ratio 0.2) | 3 | 25.3% | 23.1 | 55.3% | 30.0% | 9.2% | 81.6% |

| check | rule | result |
|---|---|---|
| clean crash | <= vanilla + 8.0% | FAILED |
| clean return | >= 0.85 x vanilla | ok |
| flip rate | <= 0.5 x vanilla | ok |
| attack-induced crashes | <= vanilla | ok |
| worst-case critic | last worst-Q of every seed in [-5, 6]: 0.06, -0.01, -0.72 | ok |

worst_ratio = 0.2, hinge_c = 0.3, kappa = 0.02, group folder = wocar_dqn_reg_b. The certified share is shown for information: WocaR's state-importance weighting leaves states with near-tied actions uncertified by design. Whether WocaR beats SA-DQN is the result of the experiment, not a gate.

Next: STOP. WocaR-DQN is not healthy and the one retry is used up. Report this table to the user; do not train anything else.

### 2026-11-15 19:39
## BLOCKED: needs the user
Gate `stage5` failed (table above). STOP. WocaR-DQN is not healthy and the one retry is used up. Report this table to the user; do not train anything else.

### 2026-11-16 00:33
User decision (made by Claude on the laptop; instruction in docs/fix3.md): continue after gate 5 FAIL. Stage 6 runs as planned with `wocar_dqn_reg_b` (worst_ratio 0.2) seeds 4-6, `sa_dqn_c030` seed 6 and `wocar_sel_dqn` seeds 1-2. Gate 5 stays FAIL; the agent records the override with `rev4.override_gate5(...)` (sets `gate5_override`, writes a RESUMED entry). Reason: WocaR is stable (worst-Q bounded), certifies 81.6% and passed every check except clean crash (25.3% vs limit 21.8%); its weaker clean driving is a result that needs 6 seeds, and `wocar_sel_dqn` tests whether our target formula causes it. Code: rev4.py (gate5_open, override_gate5), report.py (warning under the headline table when the override is set); tested on a copy of rev4_params.json.

### 2026-11-16 00:33
## RESUMED
User override of gate 5 FAIL: continue with stage 6 and wocar_dqn_reg_b despite the clean-crash FAIL (docs/fix3.md)

### 2026-11-16 00:34
Training stage 6: 0/6 runs finished
- sa_dqn_c030/model_6: failed exit code 1, last frame 0/1000000
- wocar_dqn_reg_b/model_4: failed exit code 1, last frame 0/1000000
- wocar_dqn_reg_b/model_5: failed exit code 1, last frame 0/1000000
- wocar_dqn_reg_b/model_6: failed exit code 1, last frame 0/1000000
- wocar_sel_dqn/model_1: failed exit code 1, last frame 0/1000000
- wocar_sel_dqn/model_2: failed exit code 1, last frame 0/1000000

### 2026-11-16 00:34
## BLOCKED: needs the user
6 training run(s) failed (list above). Their run.err and train.log show why; nothing was relaunched.

### 2026-11-16 00:35
Stage 6 first launch failed within 10 s on all 6 runs (system Python without torch; shell had lost the .venv). No training happened, 0 frames. Relaunching from the .venv.

### 2026-11-16 00:35
Training stage 6: 0/6 runs finished
- sa_dqn_c030/model_6: failed exit code 1, last frame 0/1000000
- wocar_dqn_reg_b/model_4: failed exit code 1, last frame 0/1000000
- wocar_dqn_reg_b/model_5: failed exit code 1, last frame 0/1000000
- wocar_dqn_reg_b/model_6: failed exit code 1, last frame 0/1000000
- wocar_sel_dqn/model_1: failed exit code 1, last frame 0/1000000
- wocar_sel_dqn/model_2: failed exit code 1, last frame 0/1000000

### 2026-11-16 00:35
## BLOCKED: needs the user
6 training run(s) failed (list above). Their run.err and train.log show why; nothing was relaunched.

### 2026-11-16 00:37
Stage 6 relaunched from the .venv: sa_dqn_c030 seed 6, wocar_dqn_reg_b seeds 4-6, wocar_sel_dqn seeds 1-2 running; CPU affinity 0x555 applied; monitor started.

### 2026-11-16 06:00
Training stage 6: 6/6 runs finished
- sa_dqn_c030/model_6: finished 
- wocar_dqn_reg_b/model_4: finished 
- wocar_dqn_reg_b/model_5: finished 
- wocar_dqn_reg_b/model_6: finished 
- wocar_sel_dqn/model_1: finished 
- wocar_sel_dqn/model_2: finished

### 2026-11-16 06:00
Stage 6 finished and committed (a884f28). Started step 9: evaluate.py --preset final, then --preset ablation (run as one managed background job).

### 2026-11-16 11:23
Evaluation preset `final`: 20 jobs done in 323 min (300 episodes, epsilons [0.0, 0.00125, 0.0025, 0.00375, 0.005, 0.0075, 0.01]).

### 2026-11-16 11:49
Evaluation preset `ablation`: 3 jobs done in 26 min (100 episodes, epsilons [0.0, 0.00125, 0.0025, 0.00375, 0.005, 0.0075, 0.01]).

### 2026-11-16 11:49
## FINISHED: rev 4 is complete
Final report: artifacts/run_1/report.html. Summary in the chat reply.

### 2026-11-19 17:50
## Stage 7 planned (user, made by Claude on the laptop; runbook docs/rev4_sel_runbook.md, branch rev4-sel)
Stage 7 trains the exploratory arm `wocar_sel_dqn` (WocaR-DQN with the authors' code's target) seeds 3-6, same settings as seeds 1-2.
Hypothesis, raised by seeds 1-2 in the final evaluation (crash without attack 16.8% against 24.3% for `wocar_dqn_reg_b`): the paper-style target is what costs WocaR its clean driving.
**Rule, fixed now, before seeds 3-6 are trained** (`gate.py sel`, `SEL_T_MIN`): after `evaluate.py --preset final` (300 episodes, every seed at the peak of the six-seed mean curve), compare the crash rate without attack of `wocar_sel_dqn` seeds 3-6 with `wocar_dqn_reg_b` (6 seeds). CONFIRMED if Welch t >= 2.0 (positive = the authors' target crashes less), otherwise NOT CONFIRMED. Seeds 1-2 are reported alongside but do not count for the rule. Informational: nothing is trained after it, and the comparison groups of the paper stay as they are.

### 2026-11-19 23:55
Stage 7 step 0 done: branch rev4-sel, imports, dry run and smoke run ok

### 2026-11-20 00:06
Stage 7 / 1c: affinity 0x555 applied to train.py processes
