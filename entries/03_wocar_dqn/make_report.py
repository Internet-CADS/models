#!/usr/bin/env python3
"""Builds the rev1 evaluation report (the same single HTML file as for rev2 and rev3) from the old notebook's cached results
and the 3rd_run training logs, instead of reading the Jupyter notebook.

Inputs (all in this folder, nothing is re-evaluated):
  artifacts/eval_cache/*.json                       results of 02_wocar_evaluation.ipynb (300-episode versions are used)
  artifacts/3rd_run/highway-v0_{vanilla,sadqn,wocar}/train.log   mini-test curves (single seed per model)
Output: artifacts/report.html

Rev1 evaluated the checkpoints picked by the safety score (return - 5 x crash rate) on the mini-test, not the peak of a mean curve,
so the robustness section is titled accordingly. Run from anywhere:  python3 entries/03_wocar_dqn/make_report.py
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT_PY = HERE.parent / "05_wocar_dqn_r3" / "report.py"
MODELS = {  # notebook model name -> (group used by report.py, 3rd_run folder)
    "Vanilla DQN": ("vanilla_dqn", "highway-v0_vanilla"),
    "SA-DQN": ("sa_dqn", "highway-v0_sadqn"),
    "WocaR-DQN (Baseline)": ("wocar_dqn", "highway-v0_wocar"),
}

def main():
    best = {}
    for f in sorted((HERE / "artifacts" / "eval_cache").glob("*.json")):
        d = json.loads(f.read_text())
        if d["model_name"] in MODELS:
            n = len(d["clean"]["episode_returns"])
            if d["model_name"] not in best or n > best[d["model_name"]][0]:
                best[d["model_name"]] = (n, d)
    missing = set(MODELS) - set(best)
    if missing:
        sys.exit(f"missing cached results for {missing}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "cache").mkdir()
        for name, (group, folder) in MODELS.items():
            n, d = best[name]
            run = tmp / "run" / group / "model_1"
            run.mkdir(parents=True)
            (run / "train.log").symlink_to(HERE / "artifacts" / "3rd_run" / folder / "train.log")
            frame = int(re.search(r"best_frame_(\d+)", d["checkpoint"]).group(1))
            payload = dict(d, group=group, seed=1, frame=frame, kind="peak", settings={"episodes": n, "pgd_steps": 10},
                           section_title="Evaluated checkpoint (best safety score on the mini-test, one seed per model)")
            (tmp / "cache" / f"{group}_model_1_frame_{frame}.json").write_text(json.dumps(payload))
        out = HERE / "artifacts" / "report.html"
        subprocess.run([sys.executable, str(REPORT_PY), str(tmp / "run"), "-o", str(out), "--eval-dir", str(tmp / "cache"),
                        "--train-eps", "0.0025", "--total-frames", "4000000", "--expected-seeds", "1",
                        "--title", "Rev 1 — first baseline (single seed per model, 4M frames)",
                        "--note", "Rev1 picked each model's checkpoint by the safety score (return - 5 x crash rate), so the curves show one run per model."],
                       check=True)
    print("wrote", out)

if __name__ == "__main__":
    main()
