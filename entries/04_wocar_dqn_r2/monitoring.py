#!/usr/bin/env python3
"""monitoring.py - live display + progress publishing for the 18 training runs (display only).

Adapted from watch_progress.py (kept in git history). Standard library only. Independent of run_orchestrator.py:
it only polls the train.log files (incrementally, by byte offset) and never signals or touches any trainer process.

    python entries/04_wocar_dqn_r2/monitoring.py                 # production view (artifacts/run_1, 4M frames)
    python entries/04_wocar_dqn_r2/monitoring.py --smoke         # artifacts/smoke/orch_flat, 3k frames
    python entries/04_wocar_dqn_r2/monitoring.py --once          # render + publish once, exit

Display: 18 rows grouped vanilla_dqn / sa_dqn / wocar_dqn, each
    run  [####------]  %  frames / total  frames/sec (smoothed)  ETA  status      (queued|training|stalled|done|failed)
Publishing: every --publish-minutes (default 15) and on exit, write PROGRESS.md and progress.json at the repo root
and commit ONLY those two files (`git commit --only`), then push the branch to origin (`--no-push` to disable; a failed push is retried on the next publish). Nothing is written or committed if the state is unchanged.
A pid lock in the git dir prevents two instances. Ctrl-C exits this script only and writes a final snapshot.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ENTRY = Path(__file__).resolve().parent
TARGET_FRAMES = 4_000_000
SMOKE_FRAMES = 3_000
N_RUNS = 6
BAR_WIDTH = 20

GROUPS = [("vanilla_dqn", "vanilla"), ("sa_dqn", "sadqn"), ("wocar_dqn", "wocar")]

FRAME_RE = re.compile(r"^frame (\d+)/(\d+)")                                   # pinned to `frame N/M`
SPEED_RE = re.compile(r"^total time:.*?speed:\s*([\d.]+)\s*frames/sec")
SPEED_ALPHA = 0.2     # EMA weight for the per-1000-frame speed readings


def repo_root(start):
    r = subprocess.run(["git", "-C", str(start), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    return Path(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() else ENTRY.parent.parent


def fmt_hms(seconds):
    if seconds is None or seconds <= 0 or seconds == float("inf"):
        return "--:--:--"
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def render_bar(frac, width=BAR_WIDTH):
    n = int(round(width * max(0.0, min(1.0, frac))))
    return "#" * n + "-" * (width - n)


class RunView:
    def __init__(self, group, model, idx, root, target):
        self.group, self.model, self.idx = group, model, idx
        self.name = f"{group}/model_{idx}"
        self.dir = root / group / f"model_{idx}"
        self.target = target
        self.log = None
        self.offset = 0
        self.partial = ""
        self.frame = 0
        self.total = target
        self.speed = None            # smoothed frames/sec
        self.mtime = None
        self.ctime = None
        self.err = None
        self.err_offset = 0
        self.crashed = False
        self.status = "queued"

    def locate(self):
        """Newest attempt's log: model_<i>/retry_<k>/... (highest k) else model_<i>/..."""
        found = None
        for k in range(0, 50):
            base = self.dir if k == 0 else self.dir / f"retry_{k}"
            p = base / "train.log"           # flat_run_dir: the run dir is the prefix itself
            if p.is_file():
                found = (p, base / "run.err")
        return found

    def _reset(self):
        self.offset, self.partial, self.frame, self.speed = 0, "", 0, None
        self.crashed, self.err_offset, self.total = False, 0, self.target

    def update(self, stall_seconds, now):
        loc = self.locate()
        if loc is None:
            self.status, self.log = "queued", None
            return
        log, err = loc
        if log != self.log:                  # first sight, or a retry attempt took over
            self.log, self.err = log, err
            self._reset()
        try:
            st = log.stat()
        except OSError:
            return
        self.mtime = st.st_mtime
        self.ctime = self.ctime or st.st_ctime
        if st.st_size < self.offset:         # truncated/replaced: start over
            self._reset()
        if st.st_size > self.offset:
            with open(log, "rb") as f:
                f.seek(self.offset)
                data = f.read()
            self.offset += len(data)
            text = self.partial + data.decode("utf-8", "replace")
            lines = text.split("\n")
            self.partial = lines.pop()       # last element is an unfinished line (or "")
            for line in lines:
                if line.startswith("frame "):
                    m = FRAME_RE.match(line)
                    if m:
                        self.frame, self.total = int(m.group(1)), int(m.group(2))
                elif line.startswith("total time:"):
                    m = SPEED_RE.match(line)
                    if m:
                        v = float(m.group(1))
                        self.speed = v if self.speed is None else SPEED_ALPHA * v + (1 - SPEED_ALPHA) * self.speed
        # tracebacks land in run.err (written by the orchestrator); scan only new bytes
        try:
            if self.err and self.err.is_file():
                sz = self.err.stat().st_size
                if sz < self.err_offset:
                    self.err_offset = 0
                if sz > self.err_offset:
                    with open(self.err, "rb") as f:
                        f.seek(self.err_offset)
                        chunk = f.read()
                    self.err_offset += len(chunk)
                    if b"Traceback (most recent call last)" in chunk:
                        self.crashed = True
        except OSError:
            pass

        if self.frame >= self.total and self.total > 0:
            self.status = "done"
        elif self.crashed:
            self.status = "failed"
        elif now - self.mtime > stall_seconds:
            self.status = "stalled"
        else:
            self.status = "training"

    @property
    def pct(self):
        return 100.0 * min(1.0, self.frame / max(1, self.total))

    @property
    def eta_seconds(self):
        if self.status != "training" or not self.speed:
            return None
        return max(0, self.total - self.frame) / self.speed


def row_text(r):
    rate = f"{r.speed:6.1f}" if r.speed and r.status in ("training", "stalled") else "     -"
    eta = fmt_hms(r.eta_seconds) if r.status == "training" else "--:--:--"
    return (f"{r.name:<20} [{render_bar(r.frame / max(1, r.total))}] {r.pct:5.1f}%  "
            f"{r.frame:>9,} / {r.total:,}  {rate} f/s  ETA {eta}  {r.status}")


def header_text(runs, now, t0):
    started = [i for i, (g, _) in enumerate(GROUPS, 1) if any(r.group == g and r.log for r in runs)]
    batch = max(started) if started else 1
    done = sum(r.status == "done" for r in runs)
    elapsed = int(now - t0) if t0 else 0
    return f"batch {batch}/{len(GROUPS)} · {done}/{len(runs)} runs done · elapsed {elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}"


def view_lines(runs, header):
    lines = [header, ""]
    for g, _ in GROUPS:
        lines += [row_text(r) for r in runs if r.group == g]
        lines.append("")
    return lines[:-1]


def snapshot(runs, header, stamp):
    return {
        "generated": stamp, "header": header, "target_frames": runs[0].target if runs else 0,
        "runs": [{"run": r.name, "group": r.group, "model_index": r.idx, "pct": round(r.pct, 2), "frames": r.frame,
                  "total": r.total, "frames_per_sec": round(r.speed, 2) if r.speed else None,
                  "eta_seconds": int(r.eta_seconds) if r.eta_seconds is not None else None,
                  "eta": fmt_hms(r.eta_seconds) if r.status == "training" else None, "status": r.status}
                 for r in runs],
    }


def signature(snap):
    """State fingerprint without timestamps/rates: publishing is skipped when this is unchanged."""
    return json.dumps([(x["run"], x["frames"], x["status"]) for x in snap["runs"]])


def markdown(runs, snap, lines):
    out = [f"# Training progress", "", f"_Generated {snap['generated']}_", "", "```text", *lines, "```", "",
           "| run | % | frames | rate (f/s) | ETA | status |", "|---|---:|---:|---:|---|---|"]
    for x in snap["runs"]:
        out.append(f"| {x['run']} | {x['pct']:.1f} | {x['frames']:,} / {x['total']:,} | "
                   f"{x['frames_per_sec'] if x['frames_per_sec'] is not None else '-'} | {x['eta'] or '-'} | {x['status']} |")
    return "\n".join(out) + "\n"


def git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True)


PUSH = True   # set from --no-push in main()


def sync_remote(repo):
    """Push local commits to origin (when enabled). Never raises: a failed push is retried on the next publish."""
    if not PUSH:
        return ""
    ahead = git(repo, "rev-list", "--count", "@{u}..HEAD")
    if ahead.returncode != 0:
        return " ; no upstream branch, not pushed"
    n = ahead.stdout.strip()
    if n == "0":
        return ""
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")   # never block on a credential prompt in a background session
    try:
        p = subprocess.run(["git", "-C", str(repo), "push", "origin", "HEAD"],
                           capture_output=True, text=True, env=env, timeout=180)
    except subprocess.TimeoutExpired:
        return " ; push timed out (retry on next publish)"
    if p.returncode == 0:
        return f" ; pushed {n} commit(s)"
    tail = (p.stderr or p.stdout).strip().splitlines()
    return f" ; push failed, retry on next publish: {tail[-1] if tail else p.returncode}"


def publish(runs, header, lines, repo, last_sig, commit, force=False):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S %z")
    snap = snapshot(runs, header, stamp)
    sig = signature(snap)
    if sig == last_sig and not force:
        return last_sig, "unchanged, skipped" + sync_remote(repo)
    md_path, js_path = repo / "PROGRESS.md", repo / "progress.json"
    md_path.write_text(markdown(runs, snap, lines), encoding="utf-8", newline="\n")
    js_path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8", newline="\n")
    if not commit:
        return sig, "written (commit disabled)"
    a = git(repo, "add", "--", "PROGRESS.md", "progress.json")
    if a.returncode != 0:
        return sig, f"written; git add failed: {a.stderr.strip()}"
    if git(repo, "diff", "--cached", "--quiet", "--", "PROGRESS.md", "progress.json").returncode == 0:
        return sig, "written; nothing to commit" + sync_remote(repo)
    c = git(repo, "commit", "--only", "-m", f"chore(progress): {stamp}", "--", "PROGRESS.md", "progress.json")
    if c.returncode != 0:
        return sig, f"written; commit failed: {(c.stderr or c.stdout).strip()}"
    return sig, "committed" + sync_remote(repo)


def pid_alive(pid):
    if os.name == "nt":   # never os.kill(pid, 0) on Windows: it terminates the process
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)      # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259      # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def acquire_lock(repo):
    gd = git(repo, "rev-parse", "--absolute-git-dir").stdout.strip()
    lock = Path(gd) / "monitoring.lock"
    if lock.exists():
        try:
            old = int(lock.read_text().strip())
        except ValueError:
            old = 0
        if old and old != os.getpid() and pid_alive(old):
            print(f"monitoring.py is already running (pid {old}); lock: {lock}")
            sys.exit(1)
    lock.write_text(str(os.getpid()))
    return lock


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", help="default artifacts/run_1 (artifacts/smoke/orch_flat with --smoke)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--target-frames", type=int)
    ap.add_argument("--repo-root", help="where PROGRESS.md/progress.json live and get committed (default: git toplevel)")
    ap.add_argument("--refresh", type=float, default=2.0, help="display refresh seconds")
    ap.add_argument("--publish-minutes", type=float, default=15.0)
    ap.add_argument("--stall-minutes", type=float, default=30.0)
    ap.add_argument("--no-commit", action="store_true", help="write progress files but never commit")
    ap.add_argument("--no-publish", action="store_true")
    ap.add_argument("--once", action="store_true", help="render and publish once, then exit")
    ap.add_argument("--no-push", action="store_true", help="commit progress locally but never push")
    args = ap.parse_args()
    global PUSH
    PUSH = not args.no_push
    try:                       # the header uses a middle dot; never crash on a narrow console code page
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

    target = args.target_frames or (SMOKE_FRAMES if args.smoke else TARGET_FRAMES)
    root = Path(args.run_root) if args.run_root else ENTRY / "artifacts" / ("smoke/orch_flat" if args.smoke else "run_1")
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    repo = Path(args.repo_root).resolve() if args.repo_root else repo_root(ENTRY)

    runs = [RunView(g, m, i, root, target) for g, m in GROUPS for i in range(1, N_RUNS + 1)]
    lock = acquire_lock(repo)
    if os.name == "nt":
        os.system("")          # enable ANSI escapes in the Windows console
    stall = args.stall_minutes * 60
    last_sig, note, last_pub = None, "", 0.0
    existing = repo / "progress.json"
    if existing.is_file():     # compare against what is already published so a restart does not re-commit
        try:
            old = json.loads(existing.read_text())
            last_sig = json.dumps([(x["run"], x["frames"], x["status"]) for x in old["runs"]])
        except (ValueError, KeyError, OSError):
            pass

    t0 = None
    try:
        if not args.once:
            print("\033[2J\033[H", end="")
        while True:
            now = time.time()
            for r in runs:
                r.update(stall, now)
            ctimes = [r.ctime for r in runs if r.ctime]
            t0 = min(ctimes) if ctimes else None
            header = header_text(runs, now, t0)
            lines = view_lines(runs, header)
            if not args.no_publish and (args.once or now - last_pub >= args.publish_minutes * 60):
                last_sig, res = publish(runs, header, lines, repo, last_sig, not args.no_commit)
                last_pub, note = now, f"last publish {time.strftime('%H:%M:%S')}: {res}"
            footer = [note, "Ctrl-C exits the monitor only (training is not touched)."] if not args.once else [note]
            out = lines + [""] + footer
            if args.once:
                print("\n".join(out))
                break
            print("\033[H" + "\n".join(x + "\033[K" for x in out) + "\033[J", end="", flush=True)
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        print("\nStopping monitor (training untouched).")
    finally:
        if not args.no_publish and not args.once:
            now = time.time()
            for r in runs:
                r.update(stall, now)
            header = header_text(runs, now, t0)
            _, res = publish(runs, header, view_lines(runs, header), repo, last_sig, not args.no_commit)
            print(f"final snapshot: {res}")
        try:
            lock.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
