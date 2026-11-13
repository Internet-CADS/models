#!/usr/bin/env python3
"""run_orchestrator.py - launch the rev 4 training runs of one stage (launching only; no display).

Run from anywhere (paths resolve from this file), with the project venv python:
    python entries/06_wocar_dqn_r4/run_orchestrator.py --stage 1             # the runs of stage 1 (all started at once)
    python entries/06_wocar_dqn_r4/run_orchestrator.py --stage 1 --dry-run   # validate, print commands, launch nothing
    python entries/06_wocar_dqn_r4/run_orchestrator.py --stage 2 --smoke     # 3k frames per run into artifacts/smoke/stage2
    python entries/06_wocar_dqn_r4/run_orchestrator.py --groups vanilla_g099 --seeds 7   # manual: any group x seeds

Stages, groups and seeds are defined in rev4.py. A stage only starts if the previous gate has passed (gate.py writes
rev4_params.json); the hyperparameters the gates decide (gamma, epsilon, hinge_c, ...) are passed to train.py as
command-line overrides from that file, so nothing is typed by hand.

Layout (relative to this entry dir):
    artifacts/run_1/<group>/model_<seed>/train.log   (+ frame_*.pth, tensorboard/, ...)
(flat: the orchestrator passes the `flat_run_dir=true` override, so train.py writes straight into --path_prefix.)

Rules: TRAIN_SEED = model index. A run is FINISHED iff it exited 0 AND its train.log reached `frame <TARGET>/<TARGET>`;
otherwise FAILED. train.py opens train.log with mode "w", so an existing train.log is NEVER launched into again: a retry
goes to model_<i>/retry_<k>/ and the old log is kept. Nothing is ever deleted, moved or committed. Standard library only.
Trainers are started in their own process group and keep running if this orchestrator is interrupted (Ctrl-C);
run the same command again to reattach (live runs are reattached, finished runs skipped).
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import rev4

ENTRY = Path(__file__).resolve().parent
TARGET_FRAMES = rev4.TARGET_FRAMES
SMOKE_FRAMES = 3_000
POLL_SECONDS = 10
ATTACH_STALE_SECONDS = 600   # an incomplete log not written for this long is treated as dead

FRAME_RE = re.compile(r"^frame (\d+)/(\d+)", re.M)                     # pinned to `frame N/M`


def now():
    return time.strftime("%H:%M:%S")


def event(msg):
    print(f"[{now()}] {msg}", flush=True)


def repo_root():
    r = subprocess.run(["git", "-C", str(ENTRY), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    return Path(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() else ENTRY.parent.parent


REPO = repo_root()


def tail_text(path, nbytes=65536):
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - nbytes))
        return f.read().decode("utf-8", "replace")


def last_frame(path):
    """None if there is no log; (0, 0) if the log has no frame line yet; else (frame, total) of the last frame line."""
    if not path.is_file():
        return None
    try:
        m = FRAME_RE.findall(tail_text(path))
    except OSError:
        return None
    return (int(m[-1][0]), int(m[-1][1])) if m else (0, 0)


class Run:
    def __init__(self, group, idx, root):
        self.group, self.idx = group, idx
        self.config = rev4.config_for(group)
        self.dir = root / group / f"model_{idx}"
        self.attempt = 0
        self.proc = None
        self.state = "pending"     # pending | running | attached | finished | failed
        self.note = ""

    @property
    def name(self):
        return f"{self.group}/model_{self.idx}"

    def prefix(self, attempt=None):
        a = self.attempt if attempt is None else attempt
        return self.dir if a == 0 else self.dir / f"retry_{a}"

    def log_path(self, attempt=None):
        return self.prefix(attempt) / "train.log"      # flat_run_dir: the run dir is the prefix itself

    def find_latest_attempt(self):
        """Highest attempt index that already has a train.log (0 if none)."""
        best = 0
        for a in range(0, 50):
            if self.log_path(a).is_file():
                best = a
        return best


def build_cmd(run, args):
    prefix = os.path.relpath(run.prefix(), ENTRY)
    cmd = [args.python, "train.py", "--config", run.config,
           "--path_prefix", prefix, "flat_run_dir=true", f"training_config:seed={run.idx}"]
    cmd += rev4.overrides(run.group, args.params)
    if args.target_frames != TARGET_FRAMES:   # only smoke/overridden runs touch num_frames
        cmd.append(f"training_config:num_frames={args.target_frames}")
    if args.smoke:
        # make a 3k-frame run reach a checkpoint, a mini-test and (robust groups) the full epsilon / worst-case ramps
        cmd += ["training_config:save_frame=3000", "training_config:mini_test=3000", "training_config:mini_test_episodes=2"]
        if rev4.role_of(run.group) != "vanilla":
            cmd += ["training_config:schedule_start=1000", "training_config:schedule_length=1000"]
        if rev4.role_of(run.group) in ("wocar", "wocar_wcl"):
            cmd.append("training_config:worst_schedule=1000")
    return cmd


def build_env():
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return env


def is_finished(path, target):
    lf = last_frame(path)
    return bool(lf and lf[0] >= target and lf[1] == target)


def launch(run, args):
    run.prefix().mkdir(parents=True, exist_ok=True)
    if run.log_path().exists():       # belt and braces: train.py would truncate it
        run.state, run.note = "failed", "refusing to launch into an existing train.log"
        event(f"run {run.name} FAILED: {run.note}")
        return
    cmd = build_cmd(run, args)
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    err = open(run.prefix() / "run.err", "ab")   # stderr only (tracebacks); stdout is already in train.log
    run.proc = subprocess.Popen(cmd, cwd=ENTRY, env=build_env(), stdout=subprocess.DEVNULL, stderr=err, **kw)
    run.state = "running"
    tag = "" if run.attempt == 0 else f" (retry {run.attempt})"
    event(f"launching run {run.name}{tag}: seed {run.idx}, pid {run.proc.pid}")


def classify(run, args):
    """Decide what to do with a run whose files may already exist (never overwrite a train.log)."""
    run.attempt = run.find_latest_attempt()
    lp = run.log_path()
    if not lp.is_file():
        return
    if is_finished(lp, args.target_frames):
        run.state, run.note = "finished", "already complete on disk"
        event(f"run {run.name} already finished (log kept)")
    elif time.time() - lp.stat().st_mtime < ATTACH_STALE_SECONDS:
        # A recently written log means a trainer is (very likely) still alive, e.g. after the orchestrator was
        # restarted. Always reattach: treating it as failed would start a second trainer on the same cores.
        run.state, run.note = "attached", "reattached to a live run via its log"
        event(f"run {run.name} reattached (log is growing)")
    else:
        lf = last_frame(lp)
        run.state = "failed"
        run.note = f"existing incomplete log at frame {lf[0] if lf else 0}; kept, not overwritten"
        event(f"run {run.name} FAILED: {run.note}")


def maybe_retry(run, args):
    if run.state == "failed" and run.attempt < args.retry:
        run.attempt += 1
        run.state, run.note = "pending", ""
        launch(run, args)


def poll(run, args):
    if run.state == "running":
        rc = run.proc.poll()
        if rc is None:
            return
        lp = run.log_path()
        if rc == 0 and is_finished(lp, args.target_frames):
            run.state = "finished"
            event(f"run {run.name} finished")
        else:
            lf = last_frame(lp)
            run.state = "failed"
            run.note = f"exit code {rc}, last frame {lf[0] if lf else 0}/{args.target_frames}"
            event(f"run {run.name} FAILED: {run.note}")
            maybe_retry(run, args)
    elif run.state == "attached":
        lp = run.log_path()
        if is_finished(lp, args.target_frames):
            run.state = "finished"
            event(f"run {run.name} finished")
        elif time.time() - lp.stat().st_mtime > ATTACH_STALE_SECONDS:
            run.state, run.note = "failed", "log stopped growing before the final frame"
            event(f"run {run.name} FAILED: {run.note}")
            maybe_retry(run, args)


def run_all(runs, args):
    for r in runs:
        classify(r, args)
    for r in runs:
        if r.state == "failed":
            maybe_retry(r, args)
    for r in runs:
        if r.state == "pending":
            launch(r, args)
            maybe_retry(r, args)
    while any(r.state in ("running", "attached") for r in runs):
        time.sleep(POLL_SECONDS)
        for r in runs:
            poll(r, args)


def dry_run(runs, args):
    ok = True
    print(f"entry dir (cwd for trainers): {ENTRY}")
    print(f"run root: {args.run_root}   target frames: {args.target_frames:,}   python: {args.python}")
    print("env per run: CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1")
    for cfg_name in sorted({r.config for r in runs}):
        cfg = ENTRY / cfg_name
        if not cfg.is_file():
            print(f"  [FAIL] missing config {cfg_name}")
            ok = False
            continue
        m = re.search(r'"num_frames":\s*(\d+)', cfg.read_text())
        nf = int(m.group(1)) if m else None
        good = nf == TARGET_FRAMES
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {cfg_name}: num_frames={nf} (target {TARGET_FRAMES:,})")
    for need in ("train.py", "defaults.json", "read_config.py", "rev4.py"):
        has = (ENTRY / need).is_file()
        ok &= has
        print(f"  [{'ok' if has else 'FAIL'}] {need} present in entry dir")
    print("\nper-run commands (nothing launched):")
    for r in runs:
        print(f"  [{r.name}] (cwd={ENTRY.name})  {' '.join(build_cmd(r, args))}")
        print(f"        log -> {os.path.relpath(r.log_path(), ENTRY)}")
    print("\nDRY RUN", "OK" if ok else "FOUND PROBLEMS")
    return 0 if ok else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, choices=[1, 2, 3, 4], help="train the runs of this stage (see rev4.py)")
    ap.add_argument("--groups", help="manual mode: comma list of group folders (see rev4.GROUPS)")
    ap.add_argument("--seeds", help="manual mode: comma list of seeds/model indices")
    ap.add_argument("--dry-run", action="store_true", help="validate, print commands, launch nothing")
    ap.add_argument("--smoke", action="store_true", help=f"{SMOKE_FRAMES} frames per run into artifacts/smoke/stage<N>, with placeholder parameters")
    ap.add_argument("--retry", type=int, default=0, help="retries per failed run (new retry_<k> dir; default 0)")
    ap.add_argument("--run-root", help="override the run root (default artifacts/run_1, or artifacts/smoke/stage<N> with --smoke)")
    ap.add_argument("--target-frames", type=int, help="override the training length (smoke default %d)" % SMOKE_FRAMES)
    ap.add_argument("--python", default=sys.executable, help="python used for train.py (default: this interpreter)")
    args = ap.parse_args()

    if bool(args.stage) == bool(args.groups):
        sys.exit("give exactly one of --stage N or --groups a,b (with --seeds)")
    args.params = rev4.load_params(smoke=args.smoke)
    if args.stage:
        pairs = rev4.stage_runs(args.stage, args.params)
    else:
        groups = args.groups.split(",")
        unknown = [g for g in groups if g not in rev4.GROUPS]
        if unknown or not args.seeds:
            sys.exit(f"unknown groups {unknown}; choose from {list(rev4.GROUPS)} and give --seeds")
        pairs = [(g, int(s)) for g in groups for s in args.seeds.split(",")]
    if args.smoke:
        args.target_frames = args.target_frames or SMOKE_FRAMES
        args.run_root = args.run_root or str(ENTRY / "artifacts" / "smoke" / f"stage{args.stage or 0}")
    args.target_frames = args.target_frames or TARGET_FRAMES
    args.run_root = Path(args.run_root) if args.run_root else rev4.RUN_ROOT
    if not args.run_root.is_absolute():
        args.run_root = (Path.cwd() / args.run_root).resolve()

    runs = [Run(g, s, args.run_root) for g, s in pairs]
    if args.dry_run:
        sys.exit(dry_run(runs, args))

    for r in runs:
        r.dir.mkdir(parents=True, exist_ok=True)
    event(f"orchestrator start: {len(runs)} runs, target {args.target_frames:,} frames, root {os.path.relpath(args.run_root, REPO)}")
    try:
        run_all(runs, args)
    except KeyboardInterrupt:
        event("interrupted: orchestrator exiting; trainers keep running (run the same command again to reattach)")
        sys.exit(130)

    print("\n=== final summary ===")
    bad = 0
    for r in runs:
        bad += r.state != "finished"
        print(f"{r.name:24s} {r.state.upper():9s} {r.note}")
    event(f"done: {len(runs) - bad}/{len(runs)} runs finished")
    if not args.smoke:
        rev4.log(f"Training{' stage ' + str(args.stage) if args.stage else ''}: {len(runs) - bad}/{len(runs)} runs finished"
                 + "".join(f"\n- {r.name}: {r.state} {r.note}" for r in runs))
        if bad:
            rev4.blocked(f"{bad} training run(s) failed (list above). Their run.err and train.log show why; "
                         "nothing was relaunched.")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
