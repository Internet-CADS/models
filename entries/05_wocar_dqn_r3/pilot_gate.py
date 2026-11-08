#!/usr/bin/env python3
"""Go / no-go check for the rev3 pilot (reads evaluate.py results + train.log, prints a verdict).

Run from entries/05_wocar_dqn_r3/ after the pilot was trained and evaluated:
    python pilot_gate.py artifacts/pilot

Rule (agreed with the user): GO if SA-DQN or WocaR has a crash rate at least GAP_NEEDED (10 points)
below vanilla, averaged over the PGD and worst-Q attacks at the training epsilon (0.005).
Sanity checks: vanilla's clean crash rate must not be very high (traffic too dense), and WocaR's
worst-critic must not have diverged.
Exit code: 0 = GO, 1 = NO-GO / needs a decision, 2 = data missing.
"""
import argparse
import re
import sys
from pathlib import Path

import report  # stdlib-only helpers shared with the HTML report

EPS_TRAIN = 0.005
EPS_EXTRA = 0.01
ATTACKS = ["pgd", "worst_q"]
GAP_NEEDED = 0.10            # crash-rate points (0.10 = 10 percentage points)
MAX_VANILLA_CLEAN_CRASH = 0.40
WORST_Q_DIVERGED = -200.0    # rev2 WocaR drifted to about -300 by 2M frames (-48 at 1M)
MIN_CLEAN_RETURN = 10.0

NUM = r"(-?[\d.]+(?:[eE][-+]?\d+)?|nan)"


def crash_at(payloads, attack, eps):
    rows = [m for e, m, _ in report.sweep_stats(payloads, attack, "crash_rate") if abs(e - eps) < 1e-12]
    return rows[0] if rows else None


def log_health(run_dir, group, seed=1):
    """(last avg batch worst q, last mini-test return, last mini-test crash rate) from train.log."""
    log = Path(run_dir) / group / f"model_{seed}" / "train.log"
    if not log.is_file():
        return None
    text = log.read_text(errors="replace")
    wq = re.findall(r"avg batch worst q:\s*" + NUM, text)
    tests = re.findall(r"this test avg reward:\s*" + NUM + r",\s*crash rate:\s*" + NUM, text)
    return (float(wq[-1]) if wq and wq[-1] != "nan" else None,
            float(tests[-1][0]) if tests else None,
            float(tests[-1][1]) if tests else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, help="pilot run folder, e.g. artifacts/pilot")
    ap.add_argument("--eval-dir", type=Path, default=None, help="default: artifacts/eval_cache/<run name>")
    ap.add_argument("--kind", default="final", help="evaluation kind to read (final / peak)")
    args = ap.parse_args()
    eval_dir = args.eval_dir or Path(__file__).resolve().parent / "artifacts" / "eval_cache" / args.run_dir.resolve().name
    if not eval_dir.is_dir():
        print(f"no evaluation results in {eval_dir}; run evaluate.py first")
        return 2
    ev = report.load_eval(eval_dir)
    van = ev.get(("vanilla_dqn", args.kind))
    if not van:
        print("no vanilla results found")
        return 2

    print(f"pilot results from {eval_dir}  (kind={args.kind})")
    van_clean = sum(d["clean"]["crash_rate"] for d in van.values()) / len(van)
    print(f"\nvanilla clean crash rate: {van_clean:.0%}" + ("   <-- very high, traffic may be too dense" if van_clean > MAX_VANILLA_CLEAN_CRASH else ""))

    best_gap, verdicts = -1.0, []
    print(f"\ncrash rate under attack (mean over {', '.join(ATTACKS)}); gap = vanilla - group, needs >= {GAP_NEEDED:.0%} at eps {EPS_TRAIN}")
    print(f"{'group':12s} {'eps':>6s} " + " ".join(f"{a:>9s}" for a in ATTACKS) + f" {'mean':>7s} {'gap':>7s}")
    for g, label in report.GROUPS:
        pl = ev.get((g, args.kind))
        if not pl:
            print(f"{label:12s} (no results)")
            continue
        for eps in (EPS_TRAIN, EPS_EXTRA):
            vals = [crash_at(pl, a, eps) for a in ATTACKS]
            vv = [crash_at(van, a, eps) for a in ATTACKS]
            if None in vals or None in vv:
                continue
            m, vm = sum(vals) / len(vals), sum(vv) / len(vv)
            gap = vm - m
            print(f"{label:12s} {eps:6g} " + " ".join(f"{v:9.0%}" for v in vals) + f" {m:7.0%} {gap:+7.0%}" + ("   (vanilla)" if g == "vanilla_dqn" else ""))
            if g != "vanilla_dqn" and eps == EPS_TRAIN:
                verdicts.append((label, gap))
                best_gap = max(best_gap, gap)

    print("\ntraining health (seed 1)")
    wocar_diverged = False
    for g, label in report.GROUPS:
        h = log_health(args.run_dir, g)
        if h is None:
            print(f"  {label}: no train.log")
            continue
        wq, ret, cr = h
        note = ""
        if g == "wocar_dqn":
            if (wq is not None and wq < WORST_Q_DIVERGED) or (ret is not None and ret < MIN_CLEAN_RETURN):
                wocar_diverged = True
                note = "   <-- WocaR looks diverged"
        print(f"  {label:10s} last mini-test return {ret if ret is not None else float('nan'):6.2f}, crash {cr if cr is not None else float('nan'):.2f}"
              + (f", avg batch worst q {wq:.1f}" if wq is not None else "") + note)

    print()
    if van_clean > MAX_VANILLA_CLEAN_CRASH:
        print("VERDICT: NO-GO / DECIDE - vanilla crashes too often even without attack. Lower env_config.vehicles_count "
              "(try 18) in all three configs and rerun the pilot; do not start the full run.")
        return 1
    if best_gap >= GAP_NEEDED:
        winners = ", ".join(f"{l} ({g:+.0%})" for l, g in verdicts if g >= GAP_NEEDED)
        msg = f"VERDICT: GO - gap over vanilla at eps {EPS_TRAIN}: {winners}."
        if wocar_diverged:
            msg += " WocaR looks diverged: in the FULL run set worst_kappa_end to 1.0 in config_highway_wocar.json first."
        print(msg)
        return 0
    print(f"VERDICT: NO-GO - best gap {best_gap:+.0%} (< {GAP_NEEDED:.0%}). Do not start the full run; report the table to the user "
          "(planned next steps: kappa 0.05, then a pgd bound_solver pilot).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
