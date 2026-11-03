#!/usr/bin/env python3
"""Live terminal progress bars for 3rd_run training logs (works on macOS & Windows)."""
import os
import re
import sys
import time

MODELS = [
    ("Vanilla DQN", "highway-v0_vanilla"),
    ("SA-DQN     ", "highway-v0_sadqn"),
    ("WocaR-DQN  ", "highway-v0_wocar"),
]

FRAME_RE = re.compile(r"^frame\s+(\d+)/(\d+),.*action epsilon:\s*([\d.eE+-]+|nan)")
PERF_RE = re.compile(
    r"total time:\s*([\d.]+),.*speed:\s*([\d.]+)\s*frames/sec,.*avg 10 episode reward:\s*([\d.eE+-]+|nan)"
)
TEST_RE = re.compile(
    r"this test avg reward:\s*([\d.eE+-]+),\s*crash rate:\s*([\d.]+),\s*safety score:\s*([\d.eE+-]+)"
)


def parse_log(log_path):
    if not os.path.isfile(log_path):
        return None
    frame, total, eps_act = 0, 4000000, "1.0"
    elapsed, speed, avg10 = 0.0, 0.0, "nan"
    test_rew, crash, safety = "-", "-", "-"

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("frame "):
                    m = FRAME_RE.match(line)
                    if m:
                        frame, total, eps_act = int(m.group(1)), int(m.group(2)), m.group(3)
                elif line.startswith("total time:"):
                    m = PERF_RE.search(line)
                    if m:
                        elapsed, speed, avg10 = float(m.group(1)), float(m.group(2)), m.group(3)
                elif line.startswith("this test avg reward:"):
                    m = TEST_RE.search(line)
                    if m:
                        test_rew, crash, safety = m.group(1), m.group(2), m.group(3)
    except OSError:
        return None

    return {
        "frame": frame,
        "total": total,
        "elapsed": elapsed,
        "speed": speed,
        "avg10": avg10,
        "test_rew": test_rew,
        "crash": crash,
        "safety": safety,
        "mtime": os.path.getmtime(log_path),
    }


def fmt_time(seconds):
    if seconds <= 0 or seconds == float("inf"):
        return "--:--:--"
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def render_bar(pct, width=28):
    filled = int(round(width * pct))
    return "█" * filled + "░" * (width - filled)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    run_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(base_dir, "artifacts", "3rd_run")

    print("\033[2J\033[H", end="")
    try:
        while True:
            lines = [f"=== 3rd Run Live Training Progress ({time.strftime('%H:%M:%S')}) ===", ""]
            found_any = False
            for label, folder in MODELS:
                log_path = os.path.join(run_dir, folder, "train.log")
                info = parse_log(log_path)
                if info is None:
                    continue
                found_any = True
                pct = min(1.0, info["frame"] / max(1, info["total"]))
                rem_frames = max(0, info["total"] - info["frame"])
                avg_fps = info["frame"] / info["elapsed"] if info["elapsed"] > 0 else info["speed"]
                eta = rem_frames / avg_fps if avg_fps > 0 else 0.0
                status = "DONE" if info["frame"] >= info["total"] else f"ETA {fmt_time(eta)}"

                bar = render_bar(pct)
                lines.append(
                    f"{label} [{bar}] {pct * 100:5.1f}%  "
                    f"{info['frame']:>9,}/{info['total']:,}  "
                    f"| {avg_fps:5.1f} FPS | Elapsed {fmt_time(info['elapsed'])} | {status}"
                )
                lines.append(
                    f"            avg10_return: {info['avg10']:>8}  "
                    f"| last mini_test -> reward: {info['test_rew']}  crash: {info['crash']}  safety: {info['safety']}"
                )
                lines.append("")

            if not found_any:
                lines.append(f"Waiting for train.log files in {run_dir} ...")

            lines.append("Press Ctrl+C to exit.")
            output = "\033[H" + "\n".join(line + "\033[K" for line in lines) + "\033[J"
            print(output, end="", flush=True)
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("\nStopped watching.")


if __name__ == "__main__":
    main()
