#!/usr/bin/env python3
"""rev4.py - shared definitions for the rev 4 scripts: stages, groups, parameters, run log. Standard library only.

Rev 4 runs in stages with a gate after each one (see ../../docs/rev4x_runbook.md):
    stage 1  vanilla DQN, two discount factors (arms) x seeds 1-3     -> gate.py stage1  (picks the arm, sets gamma + hinge_c)
             attack sweep on the chosen vanilla arm                    -> gate.py epsilon (sets the training/attack epsilon)
    stage 2  vanilla seeds 4-6 + SA-DQN seeds 1-3                      -> gate.py stage2
    stage 3  SA-DQN seeds 4-6 + WocaR-DQN seeds 1-3                    -> gate.py stage3  (informational)
    stage 4  SA-DQN with a stronger regulariser: three arms x seeds 1-2 -> gate.py reg     (picks hinge_c and kappa)
    stage 5  chosen SA-DQN seeds 3-5 + WocaR-DQN (same regulariser) seeds 1-3  -> gate.py stage5
    stage 6  chosen SA-DQN seed 6 + WocaR-DQN seeds 4-6 + exploratory arm (WocaR with the upstream target, seeds 1-2)
    final    evaluate.py --preset final, evaluate.py --preset ablation, report.py
Every stage is six runs, one per core.

Stages 4-6 were added after gate 2 (docs/rev4x_regulariser_plan.md): with hinge_c set to vanilla's action gap, SA-DQN
certified only 66% of states and crashed under attack almost as often as vanilla. The groups trained before that
(sa_dqn, wocar_dqn) are kept as the "first setting" in the regulariser ablation.

gate.py writes its decisions to rev4_params.json; run_orchestrator.py, evaluate.py, diagnose.py and report.py read them,
so nobody types hyperparameters by hand and a stage cannot start before its gate has passed.
"""
import json
import os
import time
from pathlib import Path

ENTRY = Path(__file__).resolve().parent
# The two environment variables exist for testing the pipeline without touching the real files.
PARAMS_FILE = Path(os.environ.get("REV4_PARAMS") or ENTRY / "rev4_params.json")
RUNLOG = Path(os.environ.get("REV4_RUNLOG") or ENTRY / "RUNLOG.md")
RUN_ROOT = ENTRY / "artifacts" / "run_1"
TARGET_FRAMES = 1_000_000
FIRST_SEEDS = [1, 2, 3]             # seeds a robust group starts with (the gates look at these)
LATER_SEEDS = [4, 5, 6]             # added once the group's gate has passed (6 seeds per group in the end)
ARM_SEEDS = FIRST_SEEDS             # stage 1: seeds per vanilla arm

# Stage 1 arms: vanilla DQN with different discount factors (folder -> gamma). The laptop pilot (one seed, 1M frames)
# ruled out 0.99 (17-21% crashes, slow learning); 0.97 and 0.9 both reached 10-14%.
ARMS = {"vanilla_g097": 0.97, "vanilla_g090": 0.9}

# Stage 4 arms: SA-DQN with a stronger regulariser (folder -> what differs from the first setting, i.e. from the
# hinge_c gate 1 chose and kappa 0.02). kappa weighs the regulariser against the TD loss; hinge_c is the certified
# margin up to which a state keeps being pushed. Two seeds per arm: gate.py reg selects on the certified share, which
# varies little between seeds.
REG_ARMS = {"sa_dqn_c030": {"hinge_c": 0.3}, "sa_dqn_k010": {"kappa": 0.1}, "sa_dqn_c030_k010": {"hinge_c": 0.3, "kappa": 0.1}}
REG_SEEDS = [1, 2]
# Groups trained with the regulariser gate.py reg chose (same hinge_c and kappa as the chosen SA-DQN arm).
REG_FOLLOWERS = ("wocar_dqn_reg", "wocar_dqn_reg_b", "wocar_sel_dqn", "wocar_wcl_dqn")
# Stage 6 exploratory arm (since 2026-11-15): WocaR with the authors' code's target, where the natural / worst-case
# mix only picks the next action and the natural Q value is bootstrapped. Tells whether our paper-style target
# (worst_target_mode "value") is what costs WocaR its clean driving. Replaces wocar_wcl_dqn in stage 6.
EXPLORATORY_GROUP = "wocar_sel_dqn"

# folder under the run root -> (config file, label, role)
GROUPS = {
    "vanilla_g097": ("config_highway_vanilla.json", "Vanilla DQN (gamma 0.97)", "vanilla"),
    "vanilla_g090": ("config_highway_vanilla.json", "Vanilla DQN (gamma 0.9)", "vanilla"),
    "sa_dqn": ("config_highway_sadqn.json", "SA-DQN, first setting", "sa"),
    "sa_dqn_b": ("config_highway_sadqn.json", "SA-DQN (retry)", "sa"),
    "sa_dqn_c030": ("config_highway_sadqn.json", "SA-DQN, hinge_c 0.3", "sa"),
    "sa_dqn_k010": ("config_highway_sadqn.json", "SA-DQN, kappa 0.1", "sa"),
    "sa_dqn_c030_k010": ("config_highway_sadqn.json", "SA-DQN, hinge_c 0.3 + kappa 0.1", "sa"),
    "wocar_dqn": ("config_highway_wocar.json", "WocaR-DQN, first setting", "wocar"),
    "wocar_dqn_b": ("config_highway_wocar.json", "WocaR-DQN (retry)", "wocar"),
    "wocar_dqn_reg": ("config_highway_wocar.json", "WocaR-DQN", "wocar"),
    "wocar_dqn_reg_b": ("config_highway_wocar.json", "WocaR-DQN (worst_ratio 0.2)", "wocar"),
    "wocar_sel_dqn": ("config_highway_wocar.json", "WocaR-DQN, upstream target (mix picks the action only)", "wocar"),
    "wocar_wcl_dqn": ("config_highway_wocar_wcl.json", "WocaR-DQN + worst-case loss", "wocar_wcl"),
}

DEFAULTS = {"kappa": 0.02, "worst_ratio": 0.5, "sa_group": "sa_dqn", "wocar_group": "wocar_dqn"}
# Placeholder values for --smoke runs before the gates have set the real ones (never used for a real run).
SMOKE_PARAMS = {"vanilla_group": "vanilla_g097", "gamma": 0.97, "hinge_c": 0.1, "epsilon": 0.005}

EPSILON_SEARCH = [0.0, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.05, 0.1]   # attack sweep on vanilla that picks the rev 4 epsilon


def load_params(smoke=False):
    p = dict(DEFAULTS)
    if smoke:
        p.update(SMOKE_PARAMS)
    if PARAMS_FILE.is_file():
        p.update(json.loads(PARAMS_FILE.read_text()))
    if smoke:
        p["smoke"] = True     # smoke runs skip the gate checks (need()); real values are used where the gates have set them
    return p


def save_params(p):
    p = {k: v for k, v in p.items() if k != "smoke"}
    PARAMS_FILE.write_text(json.dumps(p, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def log(text):
    """Append a timestamped block to RUNLOG.md (the human-readable progress file)."""
    stamp = time.strftime("%Y-%m-%d %H:%M")
    with open(RUNLOG, "a", encoding="utf-8", newline="\n") as f:
        f.write(f"\n### {stamp}\n{text.rstrip()}\n")


# Fixed markers the user searches for in RUNLOG.md (and in the commit messages that carry them).
BLOCKED = "BLOCKED: needs the user"
RESUMED = "RESUMED"
FINISHED = "FINISHED: rev 4 is complete"


def blocked(reason):
    log(f"## {BLOCKED}\n{reason.rstrip()}")


def resumed(note):
    log(f"## {RESUMED}\n{note.rstrip()}")


def finished(note):
    log(f"## {FINISHED}\n{note.rstrip()}")


def config_for(group):
    return GROUPS[group][0]


def schedule_end(group):
    """First frame at which a robust group trains at full strength (epsilon and worst-case ramps done); earlier
    checkpoints cannot be the evaluated peak. 0 for vanilla and unknown groups."""
    if group not in GROUPS or role_of(group) == "vanilla":
        return 0
    tc = json.loads((ENTRY / config_for(group)).read_text())["training_config"]
    end = tc["schedule_start"] + tc["schedule_length"]
    if tc.get("worst"):
        end = max(end, tc["schedule_start"] + tc["worst_schedule"])
    return end


def label_for(group):
    return GROUPS[group][1]


def role_of(group):
    return GROUPS[group][2]


def need(params, key, value, what):
    if params.get(key) != value and not params.get("smoke"):
        raise SystemExit(f"NOT ALLOWED: {what} (rev4_params.json has {key}={params.get(key)!r}, needs {value!r}). "
                         f"Follow docs/rev4x_runbook.md in order; do not edit rev4_params.json by hand.")


def decided(params, key, what):
    if params.get(key) is None and not params.get("smoke"):
        raise SystemExit(f"NOT ALLOWED: {what} (rev4_params.json has no {key}). "
                         f"Follow docs/rev4x_runbook.md in order; do not edit rev4_params.json by hand.")


def gate5_open(params, what):
    """Stage 6 and the final evaluation need gate 5 PASS, or the user's override after a FAIL (see override_gate5)."""
    if params.get("gate5") == "FAIL" and params.get("gate5_override"):
        return
    need(params, "gate5", "PASS", what)


def override_gate5(note):
    """The user's decision (2026-11-15, docs/fix3.md): continue after gate 5 FAIL with the failed WocaR group.
    gate5 stays FAIL in rev4_params.json; gate5_override records the decision."""
    params = load_params()
    if params.get("gate5") != "FAIL":
        raise SystemExit(f"NOT ALLOWED: the override is only for gate5=FAIL (rev4_params.json has {params.get('gate5')!r}).")
    params["gate5_override"] = note
    save_params(params)
    resumed(f"User override of gate 5 FAIL: {note}")


def regulariser(group, params):
    """(hinge_c, kappa) a robust group is trained with."""
    if group in REG_FOLLOWERS:
        need(params, "gate_reg", "PASS", f"training {group} before gate.py reg has passed")
        return params.get("reg_hinge_c", params["hinge_c"]), params.get("reg_kappa", params["kappa"])
    arm = REG_ARMS.get(group, {})
    return arm.get("hinge_c", params["hinge_c"]), arm.get("kappa", params["kappa"])


def overrides(group, params):
    """train.py command-line overrides for a group (the configs hold the rest)."""
    role = role_of(group)
    if role == "vanilla":
        return [f"training_config:gamma={ARMS[group]}"]
    need(params, "gate1", "PASS", "robust training before gate.py stage1 has passed")
    need(params, "gate_eps", "PASS", "robust training before gate.py epsilon has passed")
    hinge_c, kappa = regulariser(group, params)
    out = [f"training_config:gamma={params['gamma']}", f"training_config:epsilon={params['epsilon']}",
           f"training_config:hinge_c={hinge_c}", f"training_config:kappa={kappa}"]
    if role in ("wocar", "wocar_wcl"):
        out += [f"training_config:worst_eps_end={params['epsilon']}", f"training_config:worst_ratio={params['worst_ratio']}"]
    if group == "wocar_sel_dqn":
        out.append("training_config:worst_target_mode=select")
    return out


def stage_runs(stage, params):
    """[(group, seed)] that a stage trains. Raises SystemExit if the previous gate has not passed."""
    if stage == 1:
        return [(g, s) for g in ARMS for s in ARM_SEEDS]
    need(params, "gate1", "PASS", f"stage {stage} before gate.py stage1 has passed")
    need(params, "gate_eps", "PASS", f"stage {stage} before gate.py epsilon has passed")
    van, sa, woc = params["vanilla_group"], params["sa_group"], params["wocar_group"]
    if stage == 2:
        if params.get("gate2") not in (None, "RETRY") and not params.get("smoke"):
            raise SystemExit(f"NOT ALLOWED: stage 2 is already decided (gate2={params.get('gate2')}).")
        return [(van, s) for s in LATER_SEEDS] + [(sa, s) for s in FIRST_SEEDS]
    need(params, "gate2", "PASS", f"stage {stage} before gate.py stage2 has passed")
    if stage == 3:
        if params.get("gate3") not in (None, "RETRY") and not params.get("smoke"):
            raise SystemExit(f"NOT ALLOWED: stage 3 is already decided (gate3={params.get('gate3')}).")
        return [(sa, s) for s in LATER_SEEDS] + [(woc, s) for s in FIRST_SEEDS]
    decided(params, "gate3", f"stage {stage} before gate.py stage3 has run")
    if stage == 4:
        return [(g, s) for g in REG_ARMS for s in REG_SEEDS]
    need(params, "gate_reg", "PASS", f"stage {stage} before gate.py reg has passed")
    if stage == 5:
        if params.get("gate5") not in (None, "RETRY") and not params.get("smoke"):
            raise SystemExit(f"NOT ALLOWED: stage 5 is already decided (gate5={params.get('gate5')}).")
        return [(sa, s) for s in (3, 4, 5)] + [(woc, s) for s in FIRST_SEEDS]
    gate5_open(params, f"stage {stage} before gate.py stage5 has passed")
    if stage == 6:
        return [(sa, 6)] + [(woc, s) for s in LATER_SEEDS] + [(EXPLORATORY_GROUP, s) for s in REG_SEEDS]
    raise SystemExit(f"unknown stage {stage}")


def comparison_groups(params, run_root=RUN_ROOT):
    """Groups of the vanilla / SA-DQN / WocaR comparison that exist on disk, in display order."""
    names = [params.get("vanilla_group"), params.get("sa_group"), params.get("wocar_group"), EXPLORATORY_GROUP]
    return [g for g in names if g and (Path(run_root) / g).is_dir()]


def ablation_groups(params, run_root=RUN_ROOT):
    """Robust groups on disk that are not in the comparison: the first setting and the stage 4 arms that were not chosen."""
    main = comparison_groups(params, run_root)
    names = [params.get("weak_sa_group", "sa_dqn"), *REG_ARMS, params.get("weak_wocar_group", "wocar_dqn")]
    return [g for g in dict.fromkeys(names) if g not in main and (Path(run_root) / g).is_dir()]


def eval_epsilons(params):
    e = params["epsilon"]
    return [0.0] + [round(e * k, 6) for k in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)]


def preset(name, params, run_root=RUN_ROOT):
    """Evaluation presets: dict(groups, seeds, episodes, epsilons, certify, cache)."""
    if name == "epsilon":
        need(params, "gate1", "PASS", "the epsilon sweep before gate.py stage1 has passed")
        return dict(groups=[params["vanilla_group"]], seeds=ARM_SEEDS, episodes=100, epsilons=EPSILON_SEARCH, certify=False, cache="epsilon")
    need(params, "gate_eps", "PASS", f"evaluation preset {name} before gate.py epsilon has passed")
    eps = eval_epsilons(params)
    if name == "gate2":
        return dict(groups=[params["vanilla_group"], params["sa_group"]], seeds=None, episodes=100, epsilons=eps, certify=True, cache="gate")
    if name == "gate3":
        return dict(groups=[params["vanilla_group"], params["sa_group"], params["wocar_group"]], seeds=None, episodes=100, epsilons=eps, certify=True, cache="gate")
    if name == "reg":
        decided(params, "gate3", "evaluation preset reg before gate.py stage3 has run")
        return dict(groups=[params["vanilla_group"], params["sa_group"], *REG_ARMS], seeds=None, episodes=100, epsilons=eps, certify=True, cache="gate")
    need(params, "gate_reg", "PASS", f"evaluation preset {name} before gate.py reg has passed (the runbook changed on "
                                      f"2026-11-14: re-read docs/rev4x_runbook.md from step 6)")
    if name == "gate5":
        return dict(groups=[params["vanilla_group"], params["sa_group"], params["wocar_group"]], seeds=None, episodes=100, epsilons=eps, certify=True, cache="gate")
    if name == "ablation":
        return dict(groups=ablation_groups(params, run_root), seeds=None, episodes=100, epsilons=eps, certify=True, cache="gate")
    if name == "final":
        gate5_open(params, "the final evaluation before gate.py stage5 has passed")
        return dict(groups=comparison_groups(params, run_root), seeds=None, episodes=300, epsilons=eps, certify=True, cache="final")
    raise SystemExit(f"unknown preset {name}")
