#!/usr/bin/env python3
"""gate.py - the go/no-go checks of rev 4. Each gate prints a table, a verdict line, and what to do next.

    python gate.py stage1     # after stage 1 training + diagnose.py --stage1: pick the vanilla arm
    python gate.py epsilon    # after evaluate.py --preset epsilon: pick the training/attack epsilon
    python gate.py stage2     # after stage 2 training + evaluate.py --preset gate2: is SA-DQN healthy?
    python gate.py stage3     # after stage 3 training + evaluate.py --preset gate3: WocaR-DQN, first setting (informational)
    python gate.py reg        # after stage 4 training + evaluate.py --preset reg: pick the regulariser (hinge_c, kappa)
    python gate.py stage5     # after stage 5 training + evaluate.py --preset gate5: is WocaR-DQN healthy?

The verdict is PASS, RETRY (stages 2 and 5 only, once: the gate changes one hyperparameter and the stage is trained again
into a new folder) or FAIL (stop and report to the user). Decisions go to rev4_params.json, which the other scripts
read; the printed table is also appended to RUNLOG.md. Exit code: 0 = PASS, 2 = RETRY, 1 = FAIL.
A gate that has already been decided refuses to run again (--force to recompute, e.g. after fixing a broken evaluation).
Standard library only.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import report
import rev4

ENTRY = Path(__file__).resolve().parent

# --- thresholds (see docs/rev4x_runbook.md for the reasoning) -------------------------------------------------------
CRASH_MAX = 0.15          # gate 1: clean crash rate of the vanilla arm (rev 3 vanilla: 0.23)
RETURN_MIN = 18.0         # gate 1: clean return (rev 3 vanilla: 23; driving slowly in the right lane gives about 9)
Q_RATIO = (0.6, 1.5)      # gate 1: mean Q / discounted return that followed, bootstrapped at the time limit (guards
                          # against diverging values; it does not test the crash signal, see the crash-sensitivity column)
TAIL_TESTS = 3            # gate 1 looks at the last 3 mini-tests of each run (150 episodes per seed)
HINGE_RANGE = (0.02, 0.5)  # hinge_c = median best-vs-second action gap of the chosen vanilla arm, clipped to this
EPS_MIN_INCREASE = 0.20   # epsilon gate: smallest epsilon at which gradient attacks add at least this to vanilla's crash rate
EPS_MAX_TRAIN = 0.05      # ... and not larger than this (0.05 = 10 m in x, 0.8 m in y, 4 m/s)
CLEAN_CRASH_SLACK = 0.08  # gates 2/3: clean crash rate may be at most this much above vanilla's
RETURN_SHARE = 0.85       # gates 2/3: clean return at least this share of vanilla's
CERT_MIN = 0.5            # gates 2/3: certified action ratio at the training epsilon
FLIP_SHARE = 0.5          # gate 2: PGD flip rate at most this share of vanilla's
KAPPA_UP, KAPPA_DOWN = 5.0, 0.25   # gate 2 retry: regulariser too weak / too strong
WORST_RATIO_RETRY = 0.2   # gate 5 retry: weaker worst-case mixing (first attempt: 0.5)
WORST_Q_MIN = -5.0        # gates 3/5: the worst-case critic must stay bounded (a crash is -2 and ends the episode)
REG_CERT_TARGET = 0.90    # reg gate: certified share we aim for (below it the gate still picks, and says so)
REG_CERT_GAIN = 0.05      # reg gate: the chosen arm must certify at least this much more than the first setting
REG_CERT_TIE = 0.02       # reg gate: arms this close to the best certified share count as tied (lower clean crash wins)
# gate 5 does not gate WocaR's certified share: its regulariser is weighted by state importance w(s) (train.py), so
# states with near-tied actions get almost no push and stay uncertified by design (first setting: 6% vs SA-DQN 66%).
# It checks what WocaR is meant to do instead: few flips (FLIP_SHARE) and no more attack-induced crashes than vanilla.

WORST_Q_RE = re.compile(r"avg batch worst q: (-?[\d.eE+-]+|nan)")


def fmt_pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def finish(name, verdict, lines, next_step):
    text = f"**Gate `{name}`: {verdict}**\n\n" + "\n".join(lines) + f"\n\nNext: {next_step}"
    print("\n" + text + "\n")
    rev4.log(text)
    if next_step.startswith("STOP"):
        rev4.blocked(f"Gate `{name}` failed (table above). {next_step}")
    sys.exit({"PASS": 0, "NOTED": 0, "RETRY": 2}.get(verdict, 1))


def already(params, key, force):
    if params.get(key) in ("PASS", "FAIL") and not force:
        sys.exit(f"gate already decided: {key}={params[key]} (rev4_params.json). Nothing to do; continue with the runbook.")


def cache_dir(run_dir, name):
    return ENTRY / "artifacts" / "eval_cache" / Path(run_dir).name / name


def group_eval(ev, group, eps, min_seeds):
    pl = ev.get((group, "peak"))
    if not pl or len(pl) < min_seeds:
        sys.exit(f"evaluation of {group} is missing or incomplete ({len(pl) if pl else 0} seeds, need {min_seeds}); run the evaluate.py step first.")
    m = report.group_metrics(pl, eps)
    return {k: mean(v) for k, v in m.items()}, len(pl)


# ----------------------------------------------------------------------------- gate 1
def stage1(run_dir, params, force):
    already(params, "gate1", force)
    rows, stats = [], {}
    for arm, gamma in rev4.ARMS.items():
        crash, ret, ratio, gap = [], [], [], []
        for seed in rev4.ARM_SEEDS:
            log = run_dir / arm / f"model_{seed}" / "train.log"
            tests = report.parse_log(log) if log.is_file() else {}
            if not tests or max(tests) < rev4.TARGET_FRAMES:
                sys.exit(f"{arm}/model_{seed} has not reached frame {rev4.TARGET_FRAMES:,}; stage 1 training is not finished.")
            tail = [tests[f] for f in sorted(tests)[-TAIL_TESTS:]]
            diag = ENTRY / "artifacts" / "diag" / run_dir.name / f"{arm}_model_{seed}_frame_{rev4.TARGET_FRAMES}.json"
            if not diag.is_file():
                sys.exit(f"missing {diag.name}; run `python diagnose.py --stage1 --workers 6` first.")
            d = json.loads(diag.read_text())
            crash.append(mean([t[1] for t in tail]))
            ret.append(mean([t[0] for t in tail]))
            ratio.append(d["q_ratio"] if d["q_ratio"] is not None else float("inf"))
            gap.append(d["gap_median"])
            sens = "-" if d["q_before_crash"] is None else f"{(d['mean_q'] - d['q_before_crash']) / (d['mean_q'] + 2):.2f}"
            rows.append([f"{arm}/model_{seed}", gamma, fmt_pct(crash[-1]), f"{ret[-1]:.1f}", f"{ratio[-1]:.2f}",
                         "-" if d["q_before_crash"] is None else f"{d['q_before_crash']:.2f}", f"{d['mean_q']:.2f}", sens, f"{d['gap_median']:.3f}"])
        st = dict(crash=mean(crash), ret=mean(ret), ratio=mean(ratio), gap=mean(gap))
        st["ok"] = st["crash"] <= CRASH_MAX and st["ret"] >= RETURN_MIN and Q_RATIO[0] <= st["ratio"] <= Q_RATIO[1]
        stats[arm] = st
    lines = [table(["run", "gamma", "crash (last 3 mini-tests)", "return", "Q / return", "Q before crash", "mean Q", "crash sensitivity", "median action gap"], rows), "",
             table(["arm", "mean crash", "mean return", "mean Q / return", "passes"],
                   [[a, fmt_pct(s["crash"]), f"{s['ret']:.1f}", f"{s['ratio']:.2f}", "yes" if s["ok"] else "no"] for a, s in stats.items()]), "",
             f"Rule: crash <= {fmt_pct(CRASH_MAX)}, return >= {RETURN_MIN:g}, {Q_RATIO[0]} <= Q/return <= {Q_RATIO[1]}; "
             f"among passing arms the lowest crash rate wins (ties within 2 points: the higher return).",
             "Crash sensitivity = (mean Q - Q one step before a crash) / (mean Q + 2): the share of a crash's cost the "
             "value function shows just before it (information only; rev 3 vanilla: 0.20, laptop pilot: 0.30-0.45)."]
    ok = {a: s for a, s in stats.items() if s["ok"]}
    if not ok:
        params["gate1"] = "FAIL"
        rev4.save_params(params)
        finish("stage1", "FAIL", lines, "STOP. No vanilla arm is good enough. Report this table to the user; do not train anything else.")
    best = min(ok.values(), key=lambda s: s["crash"])["crash"]
    arm = max((a for a, s in ok.items() if s["crash"] <= best + 0.02), key=lambda a: ok[a]["ret"])
    hinge = round(min(max(stats[arm]["gap"], HINGE_RANGE[0]), HINGE_RANGE[1]), 3)
    params.update(gate1="PASS", vanilla_group=arm, gamma=rev4.ARMS[arm], hinge_c=hinge)
    rev4.save_params(params)
    lines.append(f"\nChosen arm: **{arm}** (gamma {rev4.ARMS[arm]}). hinge_c for the robust models: **{hinge}** "
                 f"(mean over seeds of the median action gap: {stats[arm]['gap']:.3f}, clipped to {HINGE_RANGE}).")
    finish("stage1", "PASS", lines, "`python evaluate.py --preset epsilon --workers 6`, then `python gate.py epsilon`.")


# ----------------------------------------------------------------------------- epsilon gate
def epsilon(run_dir, params, force):
    rev4.need(params, "gate1", "PASS", "the epsilon gate before gate.py stage1 has passed")
    already(params, "gate_eps", force)
    ev = report.load_eval(cache_dir(run_dir, "epsilon"))
    van = params["vanilla_group"]
    rows, chosen = [], None
    for eps in rev4.EPSILON_SEARCH[1:]:
        m, n = group_eval(ev, van, eps, len(rev4.ARM_SEEDS))
        if m.get("increase") is None:
            sys.exit(f"no attack results at epsilon {eps} for {van}; run `python evaluate.py --preset epsilon --workers 6` first.")
        usable = eps <= EPS_MAX_TRAIN and m["increase"] >= EPS_MIN_INCREASE
        if usable and chosen is None:
            chosen = eps
        rows.append([eps, f"{eps * 200:g} m / {eps * 16:g} m / {eps * 80:g} m/s", fmt_pct(m["clean_crash"]), fmt_pct(m["attack_crash"]),
                     fmt_pct(m["increase"]), fmt_pct(m.get("noise_increase")), fmt_pct(m.get("flip")), "yes" if usable else "no"])
    lines = [f"Vanilla arm {van}, {n} seeds, 100 episodes each.", "",
             table(["epsilon", "x / y / speed error", "crash, no attack", "crash, gradient attacks", "attack-induced", "random-noise-induced", "flipped (PGD)", "usable"], rows), "",
             f"Rule: the smallest epsilon <= {EPS_MAX_TRAIN} at which gradient attacks add at least {fmt_pct(EPS_MIN_INCREASE)} to vanilla's crash rate."]
    if chosen is None:
        params["gate_eps"] = "FAIL"
        rev4.save_params(params)
        finish("epsilon", "FAIL", lines, f"STOP. Vanilla is not clearly hurt by any epsilon up to {EPS_MAX_TRAIN}. Report this table to the user.")
    params.update(gate_eps="PASS", epsilon=chosen)
    rev4.save_params(params)
    lines.append(f"\nChosen epsilon: **{chosen}** (robust training and evaluation at {rev4.eval_epsilons(params)[1:]}).")
    finish("epsilon", "PASS", lines, "`python run_orchestrator.py --stage 2` (runbook step 4).")


# ----------------------------------------------------------------------------- gates 2 and 3
def health(m, van, cert_min=CERT_MIN):
    """Checks shared by the SA-DQN and WocaR gates: drives about as well as vanilla, and the regulariser took effect."""
    return {
        "clean crash": m["clean_crash"] <= van["clean_crash"] + CLEAN_CRASH_SLACK,
        "clean return": m["clean_return"] >= RETURN_SHARE * van["clean_return"],
        "certified": m.get("certified") is not None and m["certified"] >= cert_min,
    }


def compare_rows(groups, metrics, counts):
    return [[rev4.label_for(g), counts[g], fmt_pct(metrics[g]["clean_crash"]), f"{metrics[g]['clean_return']:.1f}",
             fmt_pct(metrics[g].get("attack_crash")), fmt_pct(metrics[g].get("increase")), fmt_pct(metrics[g].get("flip")),
             fmt_pct(metrics[g].get("certified"))] for g in groups]


COMPARE_HEAD = ["group", "seeds", "crash, no attack", "return, no attack", "crash, gradient attacks", "attack-induced", "flipped (PGD)", "certified"]


def stage2(run_dir, params, force):
    rev4.need(params, "gate_eps", "PASS", "gate 2 before gate.py epsilon has passed")
    already(params, "gate2", force)
    eps, van, sa = params["epsilon"], params["vanilla_group"], params["sa_group"]
    ev = report.load_eval(cache_dir(run_dir, "gate"))
    mv, nv = group_eval(ev, van, eps, 6)
    ms, ns = group_eval(ev, sa, eps, 3)
    checks = health(ms, mv)
    checks["flip rate"] = ms.get("flip") is not None and ms["flip"] <= FLIP_SHARE * mv["flip"]
    lines = [f"At the training epsilon {eps} (100 episodes per seed, peak-of-mean checkpoints).", "",
             table(COMPARE_HEAD, compare_rows([van, sa], {van: mv, sa: ms}, {van: nv, sa: ns})), "",
             table(["check", "rule", "result"], [
                 ["clean crash", f"<= vanilla + {fmt_pct(CLEAN_CRASH_SLACK)}", "ok" if checks["clean crash"] else "FAILED"],
                 ["clean return", f">= {RETURN_SHARE:g} x vanilla", "ok" if checks["clean return"] else "FAILED"],
                 ["certified", f">= {fmt_pct(CERT_MIN)}", "ok" if checks["certified"] else "FAILED"],
                 ["flip rate", f"<= {FLIP_SHARE:g} x vanilla", "ok" if checks["flip rate"] else "FAILED"]]),
             f"\nkappa = {params['kappa']}, hinge_c = {params['hinge_c']}, group folder = {sa}. "
             "The crash numbers under attack are shown for information; they are not part of this gate (3 seeds are too few)."]
    if all(checks.values()):
        params["gate2"] = "PASS"
        rev4.save_params(params)
        finish("stage2", "PASS", lines, "`python run_orchestrator.py --stage 3` (runbook step 5).")
    driving_ok = checks["clean crash"] and checks["clean return"]
    robust_ok = checks["certified"] and checks["flip rate"]
    if sa == "sa_dqn" and (driving_ok != robust_ok):
        factor = KAPPA_UP if driving_ok else KAPPA_DOWN
        why = "the regulariser is too weak (drives fine, not stable enough)" if driving_ok else "the regulariser is too strong (stable, drives too badly)"
        params.update(gate2="RETRY", kappa=round(params["kappa"] * factor, 6), sa_group="sa_dqn_b")
        rev4.save_params(params)
        lines.append(f"\nOne retry: {why}. kappa {params['kappa'] / factor:g} -> **{params['kappa']:g}**, new group folder `sa_dqn_b`.")
        finish("stage2", "RETRY", lines, "repeat runbook step 4 (`python run_orchestrator.py --stage 2` now trains `sa_dqn_b` seeds 1-3; the vanilla runs are skipped).")
    params["gate2"] = "FAIL"
    rev4.save_params(params)
    finish("stage2", "FAIL", lines, "STOP. SA-DQN is not healthy and the one retry is used up (or both kinds of check failed). Report this table to the user.")


def last_worst_q(log):
    vals = WORST_Q_RE.findall(log.read_text(errors="replace")) if log.is_file() else []
    return float(vals[-1]) if vals else float("nan")


def wocar_checks(run_dir, params, woc, mw, mv, cert_min=None):
    """cert_min None (gate 5): flip rate and attack-induced crashes replace the certified check."""
    checks = health(mw, mv, cert_min or 0.0)
    rows = [["clean crash", f"<= vanilla + {fmt_pct(CLEAN_CRASH_SLACK)}", "ok" if checks["clean crash"] else "FAILED"],
            ["clean return", f">= {RETURN_SHARE:g} x vanilla", "ok" if checks["clean return"] else "FAILED"]]
    if cert_min is None:
        del checks["certified"]
        checks["flip rate"] = mw.get("flip") is not None and mw["flip"] <= FLIP_SHARE * mv["flip"]
        checks["attack-induced"] = mw.get("increase") is not None and mw["increase"] <= mv["increase"]
        rows += [["flip rate", f"<= {FLIP_SHARE:g} x vanilla", "ok" if checks["flip rate"] else "FAILED"],
                 ["attack-induced crashes", "<= vanilla", "ok" if checks["attack-induced"] else "FAILED"]]
    else:
        rows.append(["certified", f">= {fmt_pct(cert_min)}", "ok" if checks["certified"] else "FAILED"])
    wq = [last_worst_q(run_dir / woc / f"model_{s}" / "train.log") for s in rev4.FIRST_SEEDS]
    wq_max = 0.6 / (1 - params["gamma"])
    checks["worst-case critic"] = all(math.isfinite(q) and WORST_Q_MIN <= q <= wq_max for q in wq)
    rows.append(["worst-case critic", f"last worst-Q of every seed in [{WORST_Q_MIN:g}, {wq_max:g}]: {', '.join(f'{q:.2f}' for q in wq)}",
                 "ok" if checks["worst-case critic"] else "FAILED"])
    return checks, rows


def stage3(run_dir, params, force):
    """Informational since 2026-11-14: WocaR-DQN with the first regulariser setting. Never retries, never stops."""
    rev4.need(params, "gate2", "PASS", "gate 3 before gate.py stage2 has passed")
    if params.get("gate3") is not None and not force:
        sys.exit(f"gate already decided: gate3={params['gate3']} (rev4_params.json). Nothing to do; continue with the runbook.")
    eps, van, sa, woc = params["epsilon"], params["vanilla_group"], params["sa_group"], params["wocar_group"]
    ev = report.load_eval(cache_dir(run_dir, "gate"))
    mv, nv = group_eval(ev, van, eps, 6)
    ms, ns = group_eval(ev, sa, eps, 6)
    mw, nw = group_eval(ev, woc, eps, 3)
    checks, rows = wocar_checks(run_dir, params, woc, mw, mv, CERT_MIN)
    lines = [f"At the training epsilon {eps} (100 episodes per seed, peak-of-mean checkpoints).", "",
             table(COMPARE_HEAD, compare_rows([van, sa, woc], {van: mv, sa: ms, woc: mw}, {van: nv, sa: ns, woc: nw})), "",
             table(["check", "rule", "result"], rows),
             f"\nworst_ratio = {params['worst_ratio']}, hinge_c = {params['hinge_c']}, kappa = {params['kappa']}, group folder = {woc}. "
             "This gate is informational: these groups are the first regulariser setting, kept for the ablation. "
             "The run continues with stage 4 whatever the checks say (docs/rev4x_regulariser_plan.md)."]
    verdict = "PASS" if all(checks.values()) else "NOTED"
    params["gate3"] = verdict
    rev4.save_params(params)
    finish("stage3", verdict, lines, "THE RUNBOOK CHANGED on 2026-11-14: read docs/rev4x_runbook.md again from step 6, then "
                                     "`python run_orchestrator.py --stage 4` (runbook step 6: SA-DQN with a stronger regulariser).")


def reg(run_dir, params, force):
    rev4.decided(params, "gate3", "the reg gate before gate.py stage3 has run")
    already(params, "gate_reg", force)
    eps, van = params["epsilon"], params["vanilla_group"]
    weak = params.get("weak_sa_group", params["sa_group"])
    ev = report.load_eval(cache_dir(run_dir, "gate"))
    mv, nv = group_eval(ev, van, eps, 6)
    metrics, counts = {van: mv}, {van: nv}
    metrics[weak], counts[weak] = group_eval(ev, weak, eps, 3)
    rows, ok = [], {}
    for arm in rev4.REG_ARMS:
        m, n = group_eval(ev, arm, eps, len(rev4.REG_SEEDS))
        metrics[arm], counts[arm] = m, n
        h = health(m, mv, 0.0)
        healthy = h["clean crash"] and h["clean return"]
        hinge_c, kappa = rev4.regulariser(arm, params)
        rows.append([arm, hinge_c, kappa, "ok" if h["clean crash"] else "FAILED", "ok" if h["clean return"] else "FAILED", fmt_pct(m.get("certified"))])
        if healthy and m.get("certified") is not None:
            ok[arm] = m
    groups = [van, weak, *rev4.REG_ARMS]
    lines = [f"At the training epsilon {eps} (100 episodes per seed, peak-of-mean checkpoints). "
             f"First setting ({weak}): hinge_c {params['hinge_c']}, kappa {params['kappa']}.", "",
             table(COMPARE_HEAD, compare_rows(groups, metrics, counts)), "",
             table(["arm", "hinge_c", "kappa", f"clean crash <= vanilla + {fmt_pct(CLEAN_CRASH_SLACK)}", f"clean return >= {RETURN_SHARE:g} x vanilla", "certified"], rows), "",
             f"Rule: among the arms that drive about as well as vanilla, the highest certified share wins (within "
             f"{fmt_pct(REG_CERT_TIE)}: the lower clean crash rate); it must be at least {fmt_pct(REG_CERT_GAIN)} above the "
             f"first setting. Target: {fmt_pct(REG_CERT_TARGET)}. The crash numbers under attack are shown for "
             "information; they are deliberately not part of the rule."]
    base_cert = metrics[weak].get("certified") or 0.0
    best = max((m["certified"] for m in ok.values()), default=None)
    if best is None or best < base_cert + REG_CERT_GAIN:
        params["gate_reg"] = "FAIL"
        rev4.save_params(params)
        why = "no arm drives about as well as vanilla" if best is None else "no healthy arm certifies clearly more states than the first setting"
        finish("reg", "FAIL", lines, f"STOP. A stronger regulariser did not work: {why}. Report this table to the user; do not train anything else.")
    arm = min((a for a, m in ok.items() if m["certified"] >= best - REG_CERT_TIE), key=lambda a: ok[a]["clean_crash"])
    hinge_c, kappa = rev4.regulariser(arm, params)
    params.update(gate_reg="PASS", weak_sa_group=weak, weak_wocar_group="wocar_dqn", sa_group=arm, wocar_group="wocar_dqn_reg",
                  reg_hinge_c=hinge_c, reg_kappa=kappa, worst_ratio=rev4.DEFAULTS["worst_ratio"])
    rev4.save_params(params)
    lines.append(f"\nChosen arm: **{arm}** (hinge_c **{hinge_c}**, kappa **{kappa}**, certified {fmt_pct(ok[arm]['certified'])}). "
                 f"It is now the SA-DQN group of the comparison; WocaR-DQN is trained with the same regulariser into `wocar_dqn_reg`.")
    if ok[arm]["certified"] < REG_CERT_TARGET:
        lines.append(f"Note: the chosen arm is below the {fmt_pct(REG_CERT_TARGET)} target. The run continues; mention this in the final message.")
    finish("reg", "PASS", lines, "`python run_orchestrator.py --stage 5` (runbook step 7).")


def stage5(run_dir, params, force):
    rev4.need(params, "gate_reg", "PASS", "gate 5 before gate.py reg has passed")
    already(params, "gate5", force)
    eps, van, sa, woc = params["epsilon"], params["vanilla_group"], params["sa_group"], params["wocar_group"]
    ev = report.load_eval(cache_dir(run_dir, "gate"))
    mv, nv = group_eval(ev, van, eps, 6)
    ms, ns = group_eval(ev, sa, eps, 5)
    mw, nw = group_eval(ev, woc, eps, 3)
    checks, rows = wocar_checks(run_dir, params, woc, mw, mv)
    lines = [f"At the training epsilon {eps} (100 episodes per seed, peak-of-mean checkpoints).", "",
             table(COMPARE_HEAD, compare_rows([van, sa, woc], {van: mv, sa: ms, woc: mw}, {van: nv, sa: ns, woc: nw})), "",
             table(["check", "rule", "result"], rows),
             f"\nworst_ratio = {params['worst_ratio']}, hinge_c = {params['reg_hinge_c']}, kappa = {params['reg_kappa']}, group folder = {woc}. "
             "The certified share is shown for information: WocaR's state-importance weighting leaves states with "
             "near-tied actions uncertified by design. Whether WocaR beats SA-DQN is the result of the experiment, not a gate."]
    if all(checks.values()):
        params["gate5"] = "PASS"
        rev4.save_params(params)
        finish("stage5", "PASS", lines, "`python run_orchestrator.py --stage 6` (runbook step 8).")
    if woc == "wocar_dqn_reg":
        params.update(gate5="RETRY", worst_ratio=WORST_RATIO_RETRY, wocar_group="wocar_dqn_reg_b")
        rev4.save_params(params)
        lines.append(f"\nOne retry with a weaker worst-case mixing (leans less on the worst-case value, mainly for clean "
                     f"driving): worst_ratio 0.5 -> **{WORST_RATIO_RETRY}**, new group folder `wocar_dqn_reg_b`.")
        finish("stage5", "RETRY", lines, "repeat runbook step 7 (`python run_orchestrator.py --stage 5` now trains `wocar_dqn_reg_b` seeds 1-3; the SA-DQN runs are skipped).")
    params["gate5"] = "FAIL"
    rev4.save_params(params)
    finish("stage5", "FAIL", lines, "STOP. WocaR-DQN is not healthy and the one retry is used up. Report this table to the user; do not train anything else.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gate", choices=["stage1", "epsilon", "stage2", "stage3", "reg", "stage5"])
    ap.add_argument("run_dir", type=Path, nargs="?", default=rev4.RUN_ROOT)
    ap.add_argument("--force", action="store_true", help="recompute a gate that is already decided")
    args = ap.parse_args()
    {"stage1": stage1, "epsilon": epsilon, "stage2": stage2, "stage3": stage3, "reg": reg, "stage5": stage5}[args.gate](args.run_dir.resolve(), rev4.load_params(), args.force)


if __name__ == "__main__":
    main()
