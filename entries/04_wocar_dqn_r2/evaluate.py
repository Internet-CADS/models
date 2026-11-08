#!/usr/bin/env python3
"""Robustness evaluation of the checkpoints picked by report.py (script version of the old notebook).

For each group (vanilla_dqn / sa_dqn / wocar_dqn) in a run folder, evaluate every seed at
  * "peak": the frame where the seed-mean mini-test return peaks (same rule as report.py)
  * "final": frame 4,000,000 (for comparison)
Each (group, seed, frame) is one job: clean rollouts, certified action ratio (auto_LiRPA), and
attack sweeps (random_linf / pgd / minbest / worst_q) over epsilon. Results are cached as JSON in
<cache-dir> (default artifacts/eval_cache/<run name>/), and report.py reads them.

Run from entries/04_wocar_dqn_r2/ (use the repo .venv):
    ../../.venv/bin/python evaluate.py artifacts/run_1 --episodes 100 --workers 6
Smoke test:
    ../../.venv/bin/python evaluate.py artifacts/run_1 --episodes 2 --epsilons 0,0.0025 --attacks pgd --groups vanilla_dqn --seeds 1 --frames peak
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ENTRY_DIR = Path(__file__).resolve().parent
os.chdir(ENTRY_DIR)
for sub in [ENTRY_DIR, ENTRY_DIR / "common", ENTRY_DIR / "auto_LiRPA", ENTRY_DIR / "upstream"]:
    if str(sub) not in sys.path:
        sys.path.append(str(sub))
sys.path.insert(0, str(ENTRY_DIR))

import report  # noqa: E402  (stdlib-only; peak selection shared with the HTML report)

ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
GROUP_CONFIG = {"vanilla_dqn": "config_highway_vanilla.json",
                "sa_dqn": "config_highway_sadqn.json",
                "wocar_dqn": "config_highway_wocar.json"}
FINAL_FRAME = 4_000_000
DEFAULT_EPSILONS = [0.0, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]
DEFAULT_ATTACKS = ["random_linf", "pgd", "minbest", "worst_q"]
PGD_STEPS = 10


# ----------------------------------------------------------------------------- job planning
def peak_frame(run_dir, group_dir):
    runs = report.load_group(run_dir / group_dir)
    if not runs:
        return None
    mean = report.mean_curve(runs)
    return max(mean, key=lambda p: p[1])[0]


def settings_sig(settings):
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:10]


def cache_path(cache_dir, group, seed, frame, settings):
    return cache_dir / f"{group}_model_{seed}_frame_{frame}_{settings_sig(settings)}.json"


# ----------------------------------------------------------------------------- worker side
_CTX = {}


def _ctx():
    """Lazy heavy imports (once per worker process)."""
    if not _CTX:
        import numpy as np
        import torch
        torch.set_num_threads(1)
        from highway_adapter import LegacyHighwayEnv
        from models import model_setup
        from train import get_logits_lower_bound
        _CTX.update(np=np, torch=torch, Env=LegacyHighwayEnv, model_setup=model_setup,
                    get_lb=get_logits_lower_bound)
    return _CTX


def load_q_models(ckpt_path, env_id, env):
    c = _ctx()
    torch, model_setup = c["torch"], c["model_setup"]
    sd = torch.load(ckpt_path, map_location="cpu")
    fast = model_setup(env_id, env, robust_model=False, logger=None, use_cuda=False, dueling=True, model_width=1)
    fast.features.load_state_dict(sd)
    fast.eval()
    cert = model_setup(env_id, env, robust_model=True, logger=None, use_cuda=False, dueling=True, model_width=1)
    cert.features.load_state_dict(sd)
    cert.eval()
    return fast, cert


def perturb(model, x, method, eps, rng, niters=PGD_STEPS):
    c = _ctx()
    np, torch = c["np"], c["torch"]
    nn = torch.nn
    if eps <= 0.0 or method == "clean":
        return x
    if method == "random_linf":
        return x + torch.from_numpy(rng.uniform(-eps, eps, size=x.shape).astype(np.float32))
    with torch.no_grad():
        q = model(x)
        best, worst = q.argmax(dim=1), q.argmin(dim=1)
    step = max(eps / float(niters), 1e-6)
    adv = (x + torch.from_numpy(rng.uniform(-eps, eps, size=x.shape).astype(np.float32))).clamp(x - eps, x + eps).detach()
    for _ in range(niters):
        adv.requires_grad_(True)
        q_adv = model(adv)
        if method == "pgd":
            loss = nn.CrossEntropyLoss()(q_adv, best)
        elif method == "minbest":
            mask = torch.zeros_like(q_adv).scatter_(1, best.unsqueeze(1), 1e10)
            loss = ((q_adv - mask).max(dim=1)[0] - q_adv.gather(1, best.unsqueeze(1)).squeeze(1)).sum()
        elif method == "worst_q":
            loss = (q_adv.gather(1, worst.unsqueeze(1)).squeeze(1) - q_adv.gather(1, best.unsqueeze(1)).squeeze(1)).sum()
        else:
            raise ValueError(method)
        model.zero_grad()
        loss.backward()
        adv = (adv.detach() + step * adv.grad.detach().sign()).clamp(x - eps, x + eps)
    return adv.detach()


def is_certified(cert_model, x, action, eps):
    c = _ctx()
    torch = c["torch"]
    if eps <= 0.0:
        return True
    n = cert_model.num_actions
    idx = torch.tensor([int(action)], dtype=torch.long)
    eye = torch.eye(n, dtype=x.dtype)
    cc = eye[idx].unsqueeze(1) - eye.unsqueeze(0)
    keep = ~(idx.unsqueeze(1) == torch.arange(n).unsqueeze(0))
    cc = cc[keep].view(1, n - 1, n)
    with torch.no_grad():
        lb = c["get_lb"](cert_model, x, x + eps, x - eps, eps, cc, beta=0.0)
    return bool(lb[0].min().item() > 0.0)


def run_job(job):
    """job: dict(group, seed, frame, kind, ckpt, config, out, settings). Returns (job, payload)."""
    c = _ctx()
    np, torch = c["np"], c["torch"]
    st = job["settings"]
    cfg = json.load(open(ENTRY_DIR / job["config"]))
    env_id, env_cfg = cfg["env_id"], cfg["env_config"]
    env = c["Env"](env_id, env_cfg)
    fast, cert = load_q_models(job["ckpt"], env_id, env)
    n_ep, eps_list, methods = st["episodes"], st["epsilons"], st["attacks"]
    tag = f"{job['group']}#{job['seed']}@{job['frame']}"
    t0 = time.time()

    # 1) clean rollouts
    rets, lens, crashes, states, acts, imps = [], [], [], [], [], []
    counts = {a: 0 for a in range(env.action_space.n)}
    for ep in range(n_ep):
        env.seed(ep)
        s, done, r_sum, steps, crashed = env.reset(), False, 0.0, 0, False
        while not done:
            x = torch.from_numpy(np.ascontiguousarray(s)).unsqueeze(0)
            with torch.no_grad():
                q = fast(x)[0]
            a = int(q.argmax())
            states.append(x); acts.append(a); imps.append(float(q.max() - q.min())); counts[a] += 1
            s, r, done, info = env.step(a)
            r_sum += float(r); steps += 1; crashed = crashed or bool(info.get("crashed", False))
        rets.append(r_sum); lens.append(steps); crashes.append(crashed)
    thr = float(np.percentile(imps, 75)) if imps else 0.0
    total = max(1, sum(counts.values()))

    # 2) certified action ratio on visited clean states
    cert_ratios = {}
    if st["certify"]:
        for eps in eps_list:
            cert_ratios[str(eps)] = 1.0 if eps == 0.0 else float(
                sum(is_certified(cert, x, a, eps) for x, a in zip(states, acts)) / len(states))

    # 3) attack sweeps
    sweeps = []
    for m in methods:
        for eps in eps_list:
            if eps == 0.0:
                sweeps.append(dict(method=m, epsilon=0.0, mean_return=float(np.mean(rets)), std_return=float(np.std(rets)),
                                   mean_length=float(np.mean(lens)), crash_rate=float(np.mean(crashes)),
                                   flip_rate=0.0, critical_flip_rate=0.0, noncritical_flip_rate=0.0))
                continue
            rng = np.random.RandomState(1234)
            R, L, C = [], [], []
            fl = tot = cf = cs = nf = ns = 0
            for ep in range(n_ep):
                env.seed(ep)
                s, done, r_sum, steps, crashed = env.reset(), False, 0.0, 0, False
                while not done:
                    x = torch.from_numpy(np.ascontiguousarray(s)).unsqueeze(0)
                    with torch.no_grad():
                        q = fast(x)[0]
                    a0, imp = int(q.argmax()), float(q.max() - q.min())
                    xa = perturb(fast, x, m, eps, rng)
                    with torch.no_grad():
                        a1 = int(fast(xa)[0].argmax())
                    f = int(a1 != a0)
                    fl += f; tot += 1
                    if imp >= thr: cf += f; cs += 1
                    else: nf += f; ns += 1
                    s, r, done, info = env.step(a1)
                    r_sum += float(r); steps += 1; crashed = crashed or bool(info.get("crashed", False))
                R.append(r_sum); L.append(steps); C.append(crashed)
            sweeps.append(dict(method=m, epsilon=float(eps), mean_return=float(np.mean(R)), std_return=float(np.std(R)),
                               mean_length=float(np.mean(L)), crash_rate=float(np.mean(C)),
                               flip_rate=fl / max(1, tot), critical_flip_rate=cf / max(1, cs),
                               noncritical_flip_rate=nf / max(1, ns)))
    env.close()
    payload = dict(
        group=job["group"], seed=job["seed"], frame=job["frame"], kind=job["kind"],
        checkpoint=str(Path(job["ckpt"]).relative_to(ENTRY_DIR)), settings=st, seconds=time.time() - t0,
        clean=dict(mean_return=float(np.mean(rets)), std_return=float(np.std(rets)), mean_length=float(np.mean(lens)),
                   crash_rate=float(np.mean(crashes)), mean_importance_w=float(np.mean(imps)),
                   action_fractions={ACTION_NAMES[k]: v / total for k, v in counts.items()}, episode_returns=rets),
        certified_ratios=cert_ratios, sweeps=sweeps)
    tmp = Path(job["out"]).with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(job["out"])
    return tag, payload["seconds"]


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--cache-dir", type=Path, default=None, help="default: artifacts/eval_cache/<run name>")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--epsilons", default=",".join(str(e) for e in DEFAULT_EPSILONS))
    ap.add_argument("--attacks", default=",".join(DEFAULT_ATTACKS))
    ap.add_argument("--no-certify", action="store_true")
    ap.add_argument("--groups", default=",".join(GROUP_CONFIG))
    ap.add_argument("--seeds", default="1,2,3,4,5,6")
    ap.add_argument("--frames", default="peak,final", help="comma list of: peak, final, or integer frames")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    cache_dir = (args.cache_dir or ENTRY_DIR / "artifacts" / "eval_cache" / run_dir.name).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    settings = dict(episodes=args.episodes, epsilons=[float(e) for e in args.epsilons.split(",")],
                    attacks=[a for a in args.attacks.split(",") if a], certify=not args.no_certify,
                    pgd_steps=PGD_STEPS, v=1)

    jobs = []
    for g in args.groups.split(","):
        frames = {}
        for f in args.frames.split(","):
            if f == "peak":
                pf = peak_frame(run_dir, g)
                if pf:
                    frames.setdefault(pf, "peak")
            elif f == "final":
                frames.setdefault(FINAL_FRAME, "final")
            else:
                frames.setdefault(int(f), "custom")
        for frame, kind in sorted(frames.items()):
            for seed in [int(x) for x in args.seeds.split(",")]:
                ckpt = run_dir / g / f"model_{seed}" / f"frame_{frame}.pth"
                if not ckpt.is_file():
                    print(f"skip {g}#{seed}@{frame}: checkpoint missing", file=sys.stderr)
                    continue
                out = cache_path(cache_dir, g, seed, frame, settings)
                if out.exists() and not args.force:
                    continue
                jobs.append(dict(group=g, seed=seed, frame=frame, kind=kind, ckpt=str(ckpt),
                                 config=GROUP_CONFIG[g], out=str(out), settings=settings))
    print(f"{len(jobs)} jobs to run (cache: {cache_dir})", flush=True)
    if not jobs:
        return
    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_job, j) for j in jobs]
        for fu in as_completed(futs):
            tag, secs = fu.result()
            done += 1
            print(f"[{done}/{len(jobs)}] {tag} done in {secs:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
