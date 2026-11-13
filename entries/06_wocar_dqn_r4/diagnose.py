#!/usr/bin/env python3
"""diagnose.py - is the Q-function of a checkpoint consistent, and does it register crashes?

Rev 3 trained with the time limit as an episode end, yet its Q-values (mean 43.5) were close to an endless horizon
instead of the 11.6 its episodes actually earned, and Q was still 31.7 one step before a crash.

For each (group, seed) it runs clean greedy episodes on fresh seeds (2000+, not the mini-test or evaluation seeds) and
records, per step: max Q, the gap between the best and second-best action, and the discounted return G that followed.
G ends at a crash (0 afterwards) and bootstraps with max Q at the time limit, as training does in rev 4.
- mean Q / mean G: a consistency check that catches diverging or collapsed values. At gamma >= 0.97 the bootstrapped
  tail dominates a 60-step episode, so it does NOT test the crash signal (rev 3 vanilla passes it with 1.05).
- Q one step before a crash, and crash sensitivity (mean Q - that) / (mean Q + 2): how much of a crash's cost the
  value function shows just before it (rev 3 vanilla 0.20; rev 4 laptop pilot 0.30 at gamma 0.99, 0.43-0.45 at
  0.97 / 0.9).

    python diagnose.py --stage1 --workers 6            # all stage 1 arms, final frame (what gate.py stage1 reads)
    python diagnose.py --groups sa_dqn --seeds 1,2,3   # any group; gamma comes from rev4_params.json
Results: artifacts/diag/<run name>/<group>_model_<seed>_frame_<frame>.json
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
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

import rev4  # noqa: E402

EPISODE_SEED0 = 2000


def diag_path(run_dir, group, seed, frame):
    return ENTRY_DIR / "artifacts" / "diag" / Path(run_dir).name / f"{group}_model_{seed}_frame_{frame}.json"


def run_job(job):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from highway_adapter import LegacyHighwayEnv
    from models import model_setup

    cfg = json.load(open(ENTRY_DIR / rev4.config_for(job["group"])))
    env = LegacyHighwayEnv(cfg["env_id"], cfg["env_config"])
    model = model_setup(cfg["env_id"], env, robust_model=False, logger=None, use_cuda=False, dueling=True, model_width=1)
    model.features.load_state_dict(torch.load(job["ckpt"], map_location="cpu"))
    model.eval()
    gamma = job["gamma"]

    def q_of(s):
        with torch.no_grad():
            return model(torch.from_numpy(np.ascontiguousarray(s)).unsqueeze(0))[0]

    Q, G, gaps, speeds, q_precrash, rets, crashes = [], [], [], [], [], [], 0
    counts = np.zeros(env.action_space.n)
    for ep in range(job["episodes"]):
        env.seed(EPISODE_SEED0 + ep)
        s, done, qs, rs = env.reset(), False, [], []
        while not done:
            q = q_of(s)
            top2 = torch.topk(q, 2).values
            a = int(q.argmax())
            counts[a] += 1
            qs.append(float(top2[0]))
            gaps.append(float(top2[0] - top2[1]))
            s, r, done, info = env.step(a)
            rs.append(float(r))
            speeds.append(float(env.env.unwrapped.vehicle.speed))
        ended = bool(info["terminated"])          # crash / off-road; otherwise the time limit
        crashes += bool(info["crashed"])
        rets.append(sum(rs))
        g = 0.0 if ended else float(q_of(s).max())   # bootstrap through the time limit, as training does
        for t in range(len(rs) - 1, -1, -1):
            g = rs[t] + gamma * g
            Q.append(qs[t])
            G.append(g)
        if info["crashed"]:
            q_precrash.append(qs[-1])
    env.close()
    out = dict(group=job["group"], seed=job["seed"], frame=job["frame"], gamma=gamma, episodes=job["episodes"],
               crash_rate=crashes / job["episodes"], mean_return=float(np.mean(rets)), mean_speed=float(np.mean(speeds)),
               mean_q=float(np.mean(Q)), mean_realised=float(np.mean(G)),
               q_ratio=float(np.mean(Q) / np.mean(G)) if np.mean(G) > 1e-6 else None,
               q_before_crash=float(np.mean(q_precrash)) if q_precrash else None,
               gap_p25=float(np.percentile(gaps, 25)), gap_median=float(np.median(gaps)), gap_p75=float(np.percentile(gaps, 75)),
               action_fractions={n: float(c / counts.sum()) for n, c in zip(["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"], counts)})
    path = Path(job["out"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, nargs="?", default=rev4.RUN_ROOT)
    ap.add_argument("--stage1", action="store_true", help="all stage 1 arms and seeds")
    ap.add_argument("--groups", help="comma list of group folders")
    ap.add_argument("--seeds", help="comma list of seeds (default: every model_<i> folder)")
    ap.add_argument("--frame", type=int, default=rev4.TARGET_FRAMES)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    params = rev4.load_params()
    if args.stage1:
        pairs = rev4.stage_runs(1, params)
    elif args.groups:
        pairs = []
        for g in args.groups.split(","):
            seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else sorted(
                int(p.name.split("_")[1]) for p in (run_dir / g).glob("model_*") if p.name.split("_")[1].isdigit())
            pairs += [(g, sd) for sd in seeds]
    else:
        sys.exit("give --stage1 or --groups a,b")
    jobs = []
    for g, sd in pairs:
        ckpt = run_dir / g / f"model_{sd}" / f"frame_{args.frame}.pth"
        if not ckpt.is_file():
            sys.exit(f"missing checkpoint {ckpt} (is the run finished?)")
        gamma = rev4.ARMS[g] if g in rev4.ARMS else params["gamma"]
        jobs.append(dict(group=g, seed=sd, frame=args.frame, ckpt=str(ckpt), gamma=gamma, episodes=args.episodes,
                         out=str(diag_path(run_dir, g, sd, args.frame))))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(run_job, jobs))
    print(f"{'run':<24} {'crash':>6} {'return':>7} {'speed':>6} {'mean Q':>7} {'mean G':>9} {'Q/G':>7} {'Q pre-crash':>12} {'median gap':>11}")
    for r in results:
        pc = f"{r['q_before_crash']:.2f}" if r["q_before_crash"] is not None else "-"
        ratio = f"{r['q_ratio']:.2f}" if r["q_ratio"] is not None else "-"
        print(f"{r['group'] + '/model_' + str(r['seed']):<24} {100 * r['crash_rate']:5.0f}% {r['mean_return']:7.1f} {r['mean_speed']:6.1f} "
              f"{r['mean_q']:7.2f} {r['mean_realised']:9.2f} {ratio:>7} {pc:>12} {r['gap_median']:11.3f}")


if __name__ == "__main__":
    main()
