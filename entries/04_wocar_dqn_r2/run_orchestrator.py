#!/usr/bin/env python3
"""run_orchestrator.py - launch and sequence the 18 training runs (launching only; no display).

Run from anywhere (paths resolve from this file), inside tmux, with the project venv python:
    python entries/04_wocar_dqn_r2/run_orchestrator.py              # production: 3 batches x 6 runs, 4M frames
    python entries/04_wocar_dqn_r2/run_orchestrator.py --dry-run    # precreate dirs, validate, print commands, launch nothing
    python entries/04_wocar_dqn_r2/run_orchestrator.py --smoke      # 3 batches x 6 runs, 3k frames, artifacts/smoke/orch_flat
    python entries/04_wocar_dqn_r2/run_orchestrator.py --resume     # reattach to / skip runs found on disk

Layout (relative to this entry dir):
    artifacts/run_1/{vanilla_dqn,sa_dqn,wocar_dqn}/model_{1..6}/train.log   (+ frame_*.pth, tensorboard/, ...)
(flat: the orchestrator passes the `flat_run_dir=true` override, so train.py writes straight into --path_prefix.
 A manual `train.py --config ...` without it keeps train.py's default <path_prefix>/highway-v0_<suffix>/ folders.)

Rules: batch order vanilla_dqn -> sa_dqn -> wocar_dqn, 6 concurrent trainers max, TRAIN_SEED = model index (1..6).
A run is FINISHED iff it exited 0 AND its train.log reached `frame <TARGET>/<TARGET>`; otherwise FAILED.
train.py opens train.log with mode "w", so an existing train.log is NEVER launched into again: a retry goes to
model_<i>/retry_<k>/ and the old log is kept. Nothing is ever deleted or moved. Standard library only.
Trainers are started in their own process group and keep running if this orchestrator is interrupted (Ctrl-C);
use --resume to reattach.
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ENTRY = Path(__file__).resolve().parent
TARGET_FRAMES = 4_000_000
SMOKE_FRAMES = 3_000
N_RUNS = 6            # runs per batch == max concurrent trainers (machine has 6 cores)
POLL_SECONDS = 10
ATTACH_STALE_SECONDS = 600   # --resume: an incomplete log not written for this long is treated as dead

# (folder under run root, config model name == name_suffix without underscore)
BATCHES = [("vanilla_dqn", "vanilla"), ("sa_dqn", "sadqn"), ("wocar_dqn", "wocar")]

FRAME_RE = re.compile(r"^frame (\d+)/(\d+)", re.M)                     # pinned to `frame N/M`
TEST_RE = re.compile(r"^this test avg reward: (-?[\d.eE+-]+), crash rate: ([\d.]+)", re.M)


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
    def __init__(self, key, model, idx, root):
        self.key, self.model, self.idx = key, model, idx
        self.suffix = model
        self.dir = root / key / f"model_{idx}"
        self.attempt = 0
        self.proc = None
        self.state = "pending"     # pending | running | attached | finished | failed
        self.note = ""

    @property
    def name(self):
        return f"{self.key}/model_{self.idx}"

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
    cmd = [args.python, "train.py", "--config", f"config_highway_{run.model}.json",
           "--path_prefix", prefix, "flat_run_dir=true", f"training_config:seed={run.idx}"]
    if args.target_frames != TARGET_FRAMES:   # only smoke/overridden runs touch num_frames
        cmd.append(f"training_config:num_frames={args.target_frames}")
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
        # restarted. Always reattach, with or without --resume: treating it as failed would start the next batch
        # on top of it (12 trainers on 6 cores).
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


def git(*a):
    """git with a retry when monitoring.py holds .git/index.lock at the same moment."""
    for attempt in range(10):
        r = subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True)
        if r.returncode == 0 or "index.lock" not in (r.stderr or ""):
            return r
        time.sleep(3)
    return r


def current_branch():
    return git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()


START_BRANCH = None   # set in main(); batch commits only happen on this branch


def commit_batch(runs, key, args):
    """Commit this batch's train.log + tensorboard event files (and nothing else)."""
    if START_BRANCH and current_branch() != START_BRANCH:
        event(f"WARNING: batch {key} NOT committed: branch is now '{current_branch()}', orchestrator started on "
              f"'{START_BRANCH}'. Switch back and rerun with --resume to commit it.")
        return
    files = []
    for r in runs:
        lp = r.log_path()
        if lp.is_file():
            files.append(lp)
        files += sorted((lp.parent / "tensorboard").glob("events.out.tfevents.*"))
    rel = [os.path.relpath(f, REPO) for f in files]
    if not rel:
        return
    a = git("add", "-f", "--", *rel)
    if a.returncode != 0:
        event(f"WARNING: git add failed for batch {key}: {a.stderr.strip()}")
        return
    if git("diff", "--cached", "--quiet", "--", *rel).returncode == 0:
        event(f"batch {key}: logs already committed, nothing to do")
        return
    c = git("commit", "-m", f"chore(train): {key} batch train.log + tensorboard events", "--", *rel)
    if c.returncode == 0:
        event(f"batch {key}: committed {len(rel)} files")
    else:
        event(f"WARNING: commit failed for batch {key}: {(c.stderr or c.stdout).strip()}")


def run_batch(runs, key, args):
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
    ok = sum(r.state == "finished" for r in runs)
    event(f"batch {key} complete: {ok}/{len(runs)} finished, {len(runs) - ok} failed")


def precreate(runs_by_batch):
    for runs in runs_by_batch:
        for r in runs:
            r.dir.mkdir(parents=True, exist_ok=True)


def dry_run(runs_by_batch, args):
    ok = True
    print(f"entry dir (cwd for trainers): {ENTRY}")
    print(f"run root: {args.run_root}   target frames: {args.target_frames:,}   python: {args.python}")
    print("env per run: CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1")
    for model in {m for _, m in BATCHES}:
        cfg = ENTRY / f"config_highway_{model}.json"
        if not cfg.is_file():
            print(f"  [FAIL] missing config {cfg.name}")
            ok = False
            continue
        txt = cfg.read_text()
        m = re.search(r'"num_frames":\s*(\d+)', txt)
        nf = int(m.group(1)) if m else None
        good = nf == TARGET_FRAMES
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {cfg.name}: num_frames={nf} (target {TARGET_FRAMES:,})")
    for need in ("train.py", "defaults.json", "read_config.py"):
        has = (ENTRY / need).is_file()
        ok &= has
        print(f"  [{'ok' if has else 'FAIL'}] {need} present in entry dir")
    # regex calibration against a real train.log
    cal = Path(args.calibrate_log) if args.calibrate_log else None
    if cal is None:
        cands = sorted((ENTRY / "artifacts" / "3rd_run").glob("**/train.log")) + \
                sorted((ENTRY / "artifacts" / "smoke").glob("**/train.log"))
        cal = max(cands, key=lambda p: p.stat().st_size) if cands else None   # the most complete real log
    if cal is None or not cal.is_file():
        print("  [WARN] no real train.log found to calibrate the frame/test regex against")
    else:
        txt = cal.read_text(errors="replace")
        frames = FRAME_RE.findall(txt)
        tests = TEST_RE.findall(txt)
        good = bool(frames)
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] regex calibration on {os.path.relpath(cal, REPO)}: "
              f"{len(frames)} `frame N/M` lines (last {frames[-1][0]}/{frames[-1][1]})" if frames else
              f"  [FAIL] no `frame N/M` lines in {cal}")
        print(f"        {len(tests)} `this test avg reward:` rows")
    precreate(runs_by_batch)
    print("\nper-run commands (nothing launched):")
    for runs in runs_by_batch:
        for r in runs:
            print(f"  [{r.name}] (cwd={ENTRY.name})  {' '.join(build_cmd(r, args))}")
            print(f"        log -> {os.path.relpath(r.log_path(), ENTRY)}")
    print("\nDRY RUN", "OK" if ok else "FOUND PROBLEMS")
    return 0 if ok else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="precreate folders, validate, print commands, launch nothing")
    ap.add_argument("--resume", action="store_true",
                    help="kept for compatibility: live runs (log written in the last 10 min) are now always reattached "
                         "and finished runs always skipped")
    ap.add_argument("--smoke", action="store_true", help=f"{SMOKE_FRAMES} frames per run into artifacts/smoke/orch_flat")
    ap.add_argument("--retry", type=int, default=0, help="retries per failed run (new retry_<k> dir; default 0)")
    ap.add_argument("--run-root", help="override the run root (default artifacts/run_1, or artifacts/smoke/orch_flat with --smoke)")
    ap.add_argument("--target-frames", type=int, help="override TARGET_FRAMES (smoke default %d)" % SMOKE_FRAMES)
    ap.add_argument("--python", default=sys.executable, help="python used for train.py (default: this interpreter)")
    ap.add_argument("--calibrate-log", help="real train.log to validate the regexes against (dry-run)")
    ap.add_argument("--no-commit", action="store_true", help="do not commit batch logs (default off in --smoke)")
    args = ap.parse_args()

    if args.smoke:
        args.target_frames = args.target_frames or SMOKE_FRAMES
        args.run_root = args.run_root or str(ENTRY / "artifacts" / "smoke" / "orch_flat")
        args.no_commit = True
    args.target_frames = args.target_frames or TARGET_FRAMES
    args.run_root = Path(args.run_root) if args.run_root else ENTRY / "artifacts" / "run_1"
    if not args.run_root.is_absolute():
        args.run_root = (Path.cwd() / args.run_root).resolve()

    runs_by_batch = [[Run(key, model, i, args.run_root) for i in range(1, N_RUNS + 1)] for key, model in BATCHES]
    if args.dry_run:
        sys.exit(dry_run(runs_by_batch, args))

    global START_BRANCH
    START_BRANCH = current_branch()
    precreate(runs_by_batch)
    event(f"orchestrator start (branch {START_BRANCH}): {len(BATCHES)} batches x {N_RUNS} runs, target {args.target_frames:,} frames, "
          f"root {os.path.relpath(args.run_root, REPO)}")
    try:
        for n, ((key, _), runs) in enumerate(zip(BATCHES, runs_by_batch), 1):
            event(f"starting batch {n}/{len(BATCHES)}: {key}")
            run_batch(runs, key, args)
            if not args.no_commit:
                commit_batch(runs, key, args)
    except KeyboardInterrupt:
        event("interrupted: orchestrator exiting; trainers keep running (use --resume to reattach)")
        sys.exit(130)

    print("\n=== final summary ===")
    bad = 0
    for runs in runs_by_batch:
        for r in runs:
            bad += r.state != "finished"
            print(f"{r.name:24s} {r.state.upper():9s} {r.note}")
    event(f"done: {N_RUNS * len(BATCHES) - bad}/{N_RUNS * len(BATCHES)} runs finished")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
