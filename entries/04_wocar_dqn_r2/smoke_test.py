#!/usr/bin/env python
"""Gate 3 smoke test for 04_wocar_dqn_r2 (training side only).

Run from anywhere with the project venv python:
    .venv/Scripts/python.exe entries/04_wocar_dqn_r2/smoke_test.py            # everything
    ... smoke_test.py --skip-runs                                              # re-check existing artifacts only

Steps
  1. clip check   : compute_td_loss with a tiny grad_clip -> post-clip grad norm must equal grad_clip
  2. smoke runs   : vanilla / sadqn / wocar, 60k frames, seed 1 -> artifacts/smoke/<model>/   (3 in parallel)
  3. seed check   : vanilla, seed 1, 12k frames, twice -> artifacts/smoke/seedcheck_{a,b}/  (run alongside step 2)
  4. verification : train.log parse, frame_<N>.pth cadence, no best_frame_*, mini_test row, seed repeatability

The runs go through the same CLI the orchestrator will use:
    train.py --config config_highway_<m>.json --path_prefix <dir> training_config:seed=<i> training_config:num_frames=<n>
Standard library only for steps 2-4 (step 1 imports train.py / torch). Nothing is ever deleted: existing smoke
artifacts are kept, and a run is skipped if its train.log already reaches the final frame.
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ENTRY = Path(__file__).resolve().parent
SMOKE = ENTRY / "artifacts" / "smoke"
FRAMES = 60000
SEED_FRAMES = 12000
SEED = 1

FRAME_RE = re.compile(r"^frame (\d+)/(\d+)", re.M)           # same pin as monitoring.py will use
TEST_RE = re.compile(r"this test avg reward: (-?[\d.]+), crash rate: (-?[\d.]+)")
PERF_RE = re.compile(r"^total time: [\d.]+, epoch time: [\d.]+, speed: [\d.]+ frames/sec, ", re.M)

MODELS = {"vanilla": "highway-v0_vanilla", "sadqn": "highway-v0_sadqn", "wocar": "highway-v0_wocar"}


def run_dir(prefix: Path, model: str) -> Path:
    return prefix / MODELS[model]


def final_frame(log: Path):
    if not log.exists():
        return None
    m = FRAME_RE.findall(log.read_text(errors="replace"))
    return (int(m[-1][0]), int(m[-1][1])) if m else None


def launch(model: str, prefix: Path, frames: int, seed: int):
    prefix.mkdir(parents=True, exist_ok=True)
    done = final_frame(run_dir(prefix, model) / "train.log")
    if done and done[0] >= frames - 5:
        print(f"[skip]   {model} -> {prefix.relative_to(ENTRY)} (log already at frame {done[0]}/{done[1]})")
        return None
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    cmd = [sys.executable, "train.py", "--config", f"config_highway_{model}.json",
           "--path_prefix", str(prefix.relative_to(ENTRY)),
           f"training_config:seed={seed}", f"training_config:num_frames={frames}"]
    print(f"[launch] {model} frames={frames} seed={seed} -> {prefix.relative_to(ENTRY)}")
    out = open(prefix / "stdout.txt", "a")
    return subprocess.Popen(cmd, cwd=ENTRY, env=env, stdout=out, stderr=subprocess.STDOUT)


def clip_check() -> bool:
    """Drive the real compute_td_loss with a model whose grad norm >> grad_clip; post-clip norm must equal grad_clip."""
    os.chdir(ENTRY)
    sys.path.insert(0, str(ENTRY))
    import numpy as np
    import torch
    import train

    torch.manual_seed(0)
    np.random.seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(50, 128), torch.nn.ReLU(), torch.nn.Linear(128, 5))
    tgt = torch.nn.Sequential(torch.nn.Linear(50, 128), torch.nn.ReLU(), torch.nn.Linear(128, 5))
    opt = torch.optim.SGD(net.parameters(), lr=0.0)  # lr 0: weights untouched, grads remain for inspection

    class Buf:
        def sample(self, n):
            s = np.random.randn(n, 50).astype(np.float32) * 5
            return s, np.random.randint(0, 5, n), np.random.randn(n).astype(np.float32) * 50, s.copy(), np.zeros(n, np.float32)

    class Mem:
        def get_cuda_tensors(self, *a):
            return tuple(torch.as_tensor(x) for x in a)

    results = []
    for clip in (10.0, 0.5):
        train.compute_td_loss(net, tgt, 64, Buf(), False, False, False, opt, 0.99, Mem(), False,
                              dtype=np.float32, grad_clip=clip, natural_loss_fn="huber", kappa=0.0)
        post = float(sum(p.grad.norm(2).item() ** 2 for p in net.parameters()) ** 0.5)
        results.append((clip, post))
    ok = abs(results[1][1] - 0.5) < 1e-3 and results[0][1] <= 10.0 + 1e-3
    for clip, post in results:
        print(f"[clip]   grad_clip={clip}: post-clip grad norm = {post:.4f}")
    print(f"[clip]   {'PASS' if ok else 'FAIL'} (grad_clip=0.5 must give norm 0.5; grad_clip=10 must give <= 10)")
    return ok


def norm_log(path: Path) -> list:
    """train.log from the first 'frame ' line on, with wall-clock/speed fields and timing lines removed."""
    txt = path.read_text(errors="replace")
    i = txt.find("\nframe ")
    lines = txt[i:].splitlines() if i >= 0 else []
    out = []
    for ln in lines:
        if ln.startswith("total="):  # per-section timing line
            continue
        out.append(re.sub(r"^total time: [\d.]+, epoch time: [\d.]+, speed: [\d.]+ frames/sec, ", "total time: <t>, ", ln))
    return out


def verify() -> dict:
    res = {}
    for m in MODELS:
        d = run_dir(SMOKE / m, m)
        log = d / "train.log"
        r = {"log": log.exists()}
        ff = final_frame(log)
        r["frames"] = f"{ff[0]}/{ff[1]}" if ff else "-"
        r["final_ok"] = bool(ff and ff[0] >= FRAMES - 5 and ff[1] == FRAMES)
        txt = log.read_text(errors="replace") if log.exists() else ""
        r["perf_lines"] = len(PERF_RE.findall(txt))
        tests = TEST_RE.findall(txt)
        r["mini_test"] = tests[-1] if tests else None
        ckpt = sorted(p.name for p in d.glob("frame_*.pth")) if d.exists() else []
        r["ckpts"] = ckpt
        r["ckpt_ok"] = ckpt == ["frame_50000.pth", f"frame_{ff[0] if ff else 0}.pth"] or "frame_50000.pth" in ckpt
        r["best_frame"] = sorted(p.name for p in d.glob("best_frame_*")) if d.exists() else []
        r["tb"] = bool(list(d.glob("tensorboard/events.out.tfevents.*"))) if d.exists() else False
        r["bad_words"] = [w for w in ("Traceback", "nan loss") if w in txt] + (["safety"] if "safety" in txt.lower() else [])
        res[m] = r
    a, b = SMOKE / "seedcheck_a", SMOKE / "seedcheck_b"
    la, lb = run_dir(a, "vanilla") / "train.log", run_dir(b, "vanilla") / "train.log"
    if la.exists() and lb.exists():
        na, nb = norm_log(la), norm_log(lb)
        diff = next((i for i, (x, y) in enumerate(zip(na, nb)) if x != y), None)
        res["seed"] = {"lines": (len(na), len(nb)), "identical": na == nb and len(na) > 0, "first_diff": diff,
                       "a": na[diff] if diff is not None else None, "b": nb[diff] if diff is not None else None}
    else:
        res["seed"] = {"identical": False, "note": "seedcheck logs missing"}
    return res


def report(res: dict) -> bool:
    allok = True
    print("\n=== smoke verification ===")
    for m in MODELS:
        r = res[m]
        ok = r["log"] and r["final_ok"] and r["ckpt_ok"] and not r["best_frame"] and r["tb"] and r["mini_test"] and not r["bad_words"]
        allok &= bool(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {m:8s} frames={r['frames']} perf_lines={r['perf_lines']} ckpts={r['ckpts']} "
              f"best_frame={r['best_frame']} tb={r['tb']} mini_test(reward,crash)={r['mini_test']} bad={r['bad_words']}")
    s = res["seed"]
    allok &= bool(s["identical"])
    print(f"[{'PASS' if s['identical'] else 'FAIL'}] seed repeatability (vanilla, {SEED_FRAMES} frames, seed {SEED}): {s}")
    return allok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-runs", action="store_true", help="only verify existing artifacts")
    ap.add_argument("--skip-clip", action="store_true")
    args = ap.parse_args()

    ok = True
    if not args.skip_clip:
        ok &= clip_check()
    if not args.skip_runs:
        procs = {}
        for m in MODELS:
            procs[("smoke", m)] = launch(m, SMOKE / m, FRAMES, SEED)
        procs[("seed", "a")] = launch("vanilla", SMOKE / "seedcheck_a", SEED_FRAMES, SEED)
        procs[("seed", "b")] = launch("vanilla", SMOKE / "seedcheck_b", SEED_FRAMES, SEED)
        while any(p and p.poll() is None for p in procs.values()):
            time.sleep(15)
        for k, p in procs.items():
            if p is not None:
                print(f"[exit]   {k}: returncode {p.returncode}")
                ok &= p.returncode == 0
    ok &= report(verify())
    print("\nSMOKE TEST:", "ALL PASS" if ok else "FAILURES (see above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
