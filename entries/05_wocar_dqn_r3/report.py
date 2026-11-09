#!/usr/bin/env python3
"""Build a self-contained HTML report from a run folder (e.g. artifacts/run_2).

Usage (from entries/04_wocar_dqn_r2/):
    python report.py artifacts/run_2 [-o report.html]

For each model group (vanilla_dqn, sa_dqn, wocar_dqn) it plots the mini-test
return per seed (thin lines) plus the seed-mean (bright line) with crosshair
lines at the peak of the mean curve. Data comes from `train.log`:
the `frame N/M` line preceding each `this test avg reward: <r>, crash rate: <c>`.
Pure standard library; output is a single HTML file with inline SVG.
"""
import argparse
import html
import re
import sys
from pathlib import Path

GROUPS = [("vanilla_dqn", "Vanilla DQN"), ("sa_dqn", "SA-DQN"), ("wocar_dqn", "WocaR-DQN")]
FRAME_RE = re.compile(r"^frame (\d+)/(\d+)")
TEST_RE = re.compile(r"this test avg reward:\s*(\S+),\s*crash rate:\s*(\S+)")
SEED_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#b07aa1", "#9c755f", "#76b7b2"]
MEAN_COLOR = "#e15759"


def parse_log(path):
    """Return {frame: (return, crash_rate)} from a train.log."""
    out, frame = {}, None
    with open(path, errors="replace") as f:
        for line in f:
            m = FRAME_RE.match(line)
            if m:
                frame = int(m.group(1))
                continue
            m = TEST_RE.search(line)
            if m and frame is not None:
                out[frame] = (float(m.group(1)), float(m.group(2)))
    return out


def load_group(group_dir):
    runs = {}
    for d in sorted(group_dir.glob("model_*"), key=lambda p: int(p.name.split("_")[1])):
        log = d / "train.log"
        if log.exists():
            data = parse_log(log)
            if data:
                runs[d.name] = data
    return runs


def mean_curve(runs):
    """Mean over seeds at frames where every seed has a value."""
    frames = sorted(set.intersection(*(set(r) for r in runs.values())))
    return [(fr, sum(r[fr][0] for r in runs.values()) / len(runs)) for fr in frames]


def nice_ticks(lo, hi, n=6):
    span = hi - lo or 1
    step = 10 ** (len(str(int(span / n))) - 1) if span / n >= 1 else 0.1
    for mult in (1, 2, 2.5, 5, 10):
        if span / (step * mult) <= n:
            step *= mult
            break
    t = (lo // step) * step
    ticks = []
    while t <= hi + 1e-9:
        if t >= lo - 1e-9:
            ticks.append(t)
        t += step
    return ticks


def fmt_frames(v):
    return f"{v / 1e6:g}M" if v >= 1e6 or v == 0 else f"{v / 1e3:g}k"


def chart_svg(title, runs, mean, peak, max_frame, ymax):
    W, H = 880, 420
    L, R, T, B = 60, 20, 20, 50
    pw, ph = W - L - R, H - T - B
    x = lambda fr: L + fr / max_frame * pw
    y = lambda v: T + (1 - v / ymax) * ph
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">']
    for t in nice_ticks(0, ymax):
        s.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>'
                 f'<text class="tick" x="{L - 8}" y="{y(t) + 4:.1f}" text-anchor="end">{t:g}</text>')
    for t in nice_ticks(0, max_frame, 8):
        s.append(f'<text class="tick" x="{x(t):.1f}" y="{H - B + 18}" text-anchor="middle">{fmt_frames(t)}</text>')
    s.append(f'<text class="axis" x="{L + pw / 2}" y="{H - 8}" text-anchor="middle">training frames</text>')
    s.append(f'<text class="axis" transform="translate(14 {T + ph / 2}) rotate(-90)" text-anchor="middle">mini-test return</text>')
    for i, (name, data) in enumerate(runs.items()):
        pts = " ".join(f"{x(fr):.1f},{y(v[0]):.1f}" for fr, v in sorted(data.items()))
        s.append(f'<polyline class="seed" stroke="{SEED_COLORS[i % 6]}" points="{pts}"><title>{name}</title></polyline>')
    # peak crosshair
    pf, pv = peak
    s.append(f'<line class="peak" x1="{x(pf):.1f}" x2="{x(pf):.1f}" y1="{T}" y2="{T + ph}"/>'
             f'<line class="peak" x1="{L}" x2="{W - R}" y1="{y(pv):.1f}" y2="{y(pv):.1f}"/>')
    pts = " ".join(f"{x(fr):.1f},{y(v):.1f}" for fr, v in mean)
    s.append(f'<polyline class="mean" points="{pts}"><title>mean over seeds</title></polyline>')
    s.append(f'<circle cx="{x(pf):.1f}" cy="{y(pv):.1f}" r="5" fill="{MEAN_COLOR}" stroke="#fff" stroke-width="1.5"/>')
    lx = min(max(x(pf), L + 70), W - R - 70)
    s.append(f'<text class="peaklbl" x="{lx:.1f}" y="{T + 12}" text-anchor="middle">peak {pv:.2f} @ {fmt_frames(pf)}</text>')
    s.append("</svg>")
    return "".join(s)

# ----------------------------------------------------------------------------- robustness section
GROUP_COLORS = {"vanilla_dqn": "#4e79a7", "sa_dqn": "#f28e2b", "wocar_dqn": "#59a14f"}
ATTACKS = [("random_linf", "Random L-inf noise"), ("pgd", "10-step PGD"),
           ("minbest", "MinBest"), ("worst_q", "Worst-action (min-Q) PGD")]
TRAIN_EPS = 0.005          # rev3 training epsilon (config_highway_sadqn/wocar.json)
TOTAL_FRAMES = 2_000_000  # rev3 training length
EXPECTED_SEEDS = 3        # runs per group in rev3


def load_eval(cache_dir):
    """{(group, kind): {seed: payload}} using the newest JSON per (group, seed, frame)."""
    import json
    newest = {}
    for f in sorted(cache_dir.glob("*.json"), key=lambda p: p.stat().st_mtime):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        d["_mtime"] = f.stat().st_mtime
        key = (d["group"], d["seed"], d["frame"])
        prev = newest.get(key)
        # Same checkpoint evaluated in several runs (e.g. extra epsilons added later): merge results.
        # Only merge when episode count / PGD steps agree; newer entries win on (method, eps) clashes.
        if prev and all(prev["settings"].get(k) == d["settings"].get(k) for k in ("episodes", "pgd_steps")):
            sw = {(r["method"], r["epsilon"]): r for r in prev["sweeps"]}
            sw.update({(r["method"], r["epsilon"]): r for r in d["sweeps"]})
            d["sweeps"] = sorted(sw.values(), key=lambda r: (r["method"], r["epsilon"]))
            d["certified_ratios"] = {**prev["certified_ratios"], **d["certified_ratios"]}
        newest[key] = d
    out, stamp = {}, {}
    for (g, seed, frame), d in newest.items():
        # if a group was evaluated at several "peak" frames (e.g. provisional, then final), keep the newest
        t = d.get("_mtime", 0)
        k = (g, d["kind"], seed)
        if k not in stamp or t >= stamp[k]:
            stamp[k] = t
            out.setdefault((g, d["kind"]), {})[seed] = d
    return out


def _ms(vals):
    n = len(vals)
    m = sum(vals) / n
    sd = (sum((v - m) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return m, sd


def sweep_stats(payloads, method, key):
    """[(eps, mean, std)] over seeds."""
    by_eps = {}
    for d in payloads.values():
        for r in d["sweeps"]:
            if r["method"] == method:
                by_eps.setdefault(r["epsilon"], []).append(r[key])
    return [(e, *_ms(v)) for e, v in sorted(by_eps.items())]


def eps_chart(title, series, ylabel, ylim=None, pct=False, train_eps=TRAIN_EPS):
    """series: [(label, color, [(eps, mean, std)], dashed)]; x is categorical (evenly spaced epsilons)."""
    W, H, L, R, T, B = 330, 250, 46, 10, 26, 44
    eps_all = sorted({e for _, _, pts, _ in series for e, _, _ in pts})
    if not eps_all:
        return ""
    k = 100.0 if pct else 1.0
    hi = max([m * k + sd * k for _, _, pts, _ in series for _, m, sd in pts] + [1e-9])
    lo = min([m * k - sd * k for _, _, pts, _ in series for _, m, sd in pts] + [0.0])
    y0, y1 = ylim if ylim else (min(lo, 0.0), hi * 1.1)
    px = lambda e: L + eps_all.index(e) / max(1, len(eps_all) - 1) * (W - L - R)
    py = lambda v: T + (1 - (v - y0) / (y1 - y0 or 1)) * (H - T - B)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">',
         f'<text class="ctitle" x="{L}" y="15">{html.escape(title)}</text>']
    for t in nice_ticks(y0, y1, 5):
        s.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{py(t):.1f}" y2="{py(t):.1f}"/>'
                 f'<text class="tick" x="{L - 6}" y="{py(t) + 4:.1f}" text-anchor="end">{t:g}</text>')
    for e in eps_all:
        s.append(f'<text class="tick" x="{px(e):.1f}" y="{H - B + 14}" text-anchor="middle" '
                 f'transform="rotate(35 {px(e):.1f} {H - B + 14})">{e:g}</text>')
    if train_eps in eps_all:
        s.append(f'<line class="trainEps" x1="{px(train_eps):.1f}" x2="{px(train_eps):.1f}" y1="{T}" y2="{H - B}"/>')
    s.append(f'<text class="axis" x="{L + (W - L - R) / 2}" y="{H - 4}" text-anchor="middle">epsilon</text>')
    s.append(f'<text class="axis" transform="translate(11 {T + (H - T - B) / 2}) rotate(-90)" text-anchor="middle">{html.escape(ylabel)}</text>')
    for label, color, pts, dashed in series:
        if len(pts) > 0 and any(sd > 0 for _, _, sd in pts):
            top = " ".join(f"{px(e):.1f},{py((m + sd) * k):.1f}" for e, m, sd in pts)
            bot = " ".join(f"{px(e):.1f},{py((m - sd) * k):.1f}" for e, m, sd in reversed(pts))
            s.append(f'<polygon points="{top} {bot}" fill="{color}" opacity=".15"/>')
        line = " ".join(f"{px(e):.1f},{py(m * k):.1f}" for e, m, _ in pts)
        s.append(f'<polyline class="eline" stroke="{color}" points="{line}"'
                 + (' stroke-dasharray="5 3"' if dashed else "") + f'><title>{html.escape(label)}</title></polyline>')
        s += [f'<circle cx="{px(e):.1f}" cy="{py(m * k):.1f}" r="2.6" fill="{color}"><title>{html.escape(label)} eps={e:g}: {m * k:.2f} &plusmn; {sd * k:.2f}</title></circle>'
              for e, m, sd in pts]
    s.append("</svg>")
    return "".join(s)


def group_legend(present):
    return '<div class="legend">' + "".join(
        f'<span><i style="background:{GROUP_COLORS[g]}"></i>{label}</span>' for g, label in present
    ) + f'<span><i style="background:#d33;height:1px"></i>training epsilon ({TRAIN_EPS:g})</span></div>'


def robustness_section(ev, kind):
    present = [(g, label) for g, label in GROUPS if (g, kind) in ev]
    if not present:
        return ""
    desc = []
    for g, label in present:
        pl = ev[(g, kind)]
        frame = next(iter(pl.values()))["frame"]
        d = next(iter(pl.values()))
        desc.append(f"{label}: frame {frame:,} ({len(pl)} seeds, {d['settings']['episodes']} episodes)")
    fr0 = next(iter(ev[(present[0][0], kind)].values()))["frame"]
    title = "Peak-of-mean checkpoint" if kind == "peak" else f"Final checkpoint ({fr0 / 1e6:g}M frames)"
    sec = [f'<h1 class="sect">Robustness &mdash; {title}</h1><p class="sub">{"; ".join(desc)}. '
           f'Lines = mean over seeds, band = &plusmn;1 std over seeds. Evaluated on episode seeds 0&ndash;N&minus;1 (not the mini-test seeds).</p>']

    # tables at eps_train and 2x
    for eps in (TRAIN_EPS, TRAIN_EPS * 2):
        head = "<tr><th>group</th><th>natural return</th><th>natural crash</th>" + "".join(
            f"<th>{n}<br>return / crash</th>" for _, n in ATTACKS) + "<th>certified ratio</th></tr>"
        rows = ""
        for g, label in present:
            pl = ev[(g, kind)]
            nr = _ms([d["clean"]["mean_return"] for d in pl.values()])
            nc = _ms([d["clean"]["crash_rate"] for d in pl.values()])
            cells = f"<td>{nr[0]:.2f} &plusmn; {nr[1]:.2f}</td><td>{nc[0] * 100:.1f}% &plusmn; {nc[1] * 100:.1f}</td>"
            for m, _ in ATTACKS:
                r = [x for x in sweep_stats(pl, m, "mean_return") if abs(x[0] - eps) < 1e-12]
                c = [x for x in sweep_stats(pl, m, "crash_rate") if abs(x[0] - eps) < 1e-12]
                cells += (f"<td>{r[0][1]:.2f} &plusmn; {r[0][2]:.2f} / {c[0][1] * 100:.0f}%</td>" if r and c else "<td>&ndash;</td>")
            cr = [d["certified_ratios"].get(str(eps)) for d in pl.values() if d["certified_ratios"].get(str(eps)) is not None]
            cells += f"<td>{_ms(cr)[0] * 100:.1f}%</td>" if cr else "<td>&ndash;</td>"
            rows += f'<tr><td><b style="color:{GROUP_COLORS[g]}">{label}</b></td>{cells}</tr>'
        sec.append(f'<section class="card"><h2>At epsilon = {eps:g}{" (training budget)" if eps == TRAIN_EPS else " (2&times; training budget)"}</h2>'
                   f'<div class="scroll"><table class="data">{head}{rows}</table></div></section>')

    # attack curves
    for m, mname in ATTACKS:
        if not any(sweep_stats(ev[(g, kind)], m, "flip_rate") for g, _ in present):
            continue   # attack not evaluated
        charts = []
        for key, ylabel, pct, ylim in [("mean_return", "return", False, None),
                                       ("crash_rate", "crash rate (%)", True, (0, 100)),
                                       ("flip_rate", "action flipped (%)", True, (0, 100))]:
            series = [(label, GROUP_COLORS[g], sweep_stats(ev[(g, kind)], m, key), False) for g, label in present]
            charts.append(eps_chart(ylabel if key != "mean_return" else "mean return", series, ylabel, ylim, pct))
        sec.append(f'<section class="card"><h2>{mname}</h2><div class="triple">{"".join(charts)}</div>{group_legend(present)}</section>')

    # certified vs empirical
    cert_series = []
    for g, label in present:
        pl = ev[(g, kind)]
        by = {}
        for d in pl.values():
            for e, v in d["certified_ratios"].items():
                by.setdefault(float(e), []).append(v)
        if by:
            cert_series.append((f"{label} certified", GROUP_COLORS[g], [(e, *_ms(v)) for e, v in sorted(by.items())], False))
        wq = sweep_stats(pl, "worst_q", "flip_rate")
        if wq:
            cert_series.append((f"{label} empirical unflipped (worst-Q PGD)", GROUP_COLORS[g], [(e, 1 - m, sd) for e, m, sd in wq], True))
    if cert_series:
        sec.append('<section class="card"><h2>Certified vs. empirical action stability</h2>'
                   '<p class="sub">Solid = share of visited states whose action is provably unchanged (auto_LiRPA). '
                   'Dashed = share not flipped by the worst-Q PGD attack. Certified must lie below dashed.</p>'
                   f'<div class="triple">{eps_chart("unchanged action (%)", cert_series, "% of steps", (0, 105), True)}</div>'
                   f'{group_legend(present)}</section>')
    return "".join(sec)


CSS = """
:root{--bg:#fff;--fg:#1f2328;--muted:#6b7280;--grid:#e5e7eb;--card:#fff;--bd:#e5e7eb}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e6e8eb;--muted:#9aa1ab;--grid:#2c3037;--card:#1b1e23;--bd:#2c3037}}
body{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
main{max-width:940px;margin:auto}h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:0 0 4px}
.sub{color:var(--muted);margin:0 0 20px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:16px;margin:0 0 20px}
svg{width:100%;height:auto;display:block}
.grid{stroke:var(--grid)}.tick{fill:var(--muted);font-size:11px}.axis{fill:var(--muted);font-size:12px}
.seed{fill:none;stroke-width:1.2;opacity:.55}
.mean{fill:none;stroke:@M@;stroke-width:3.2;stroke-linejoin:round}
.peak{stroke:@M@;stroke-width:1;stroke-dasharray:5 4;opacity:.8}
.peaklbl{fill:@M@;font-size:12px;font-weight:600}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;color:var(--muted);font-size:12px;margin-top:6px}
.legend i{display:inline-block;width:14px;height:3px;margin-right:5px;vertical-align:middle}
table{border-collapse:collapse;font-size:13px}td,th{padding:3px 12px 3px 0;text-align:left}th{color:var(--muted);font-weight:500}
.warn{color:#b45309}
h1.sect{font-size:20px;margin:36px 0 4px;padding-top:16px;border-top:1px solid var(--bd)}
.triple{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:8px}
.ctitle{fill:var(--fg);font-size:12px;font-weight:600}
.trainEps{stroke:#d33;stroke-width:1;stroke-dasharray:3 3;opacity:.6}
.eline{fill:none;stroke-width:2}
.scroll{overflow-x:auto}table.data td,table.data th{padding:5px 14px 5px 0;white-space:nowrap}
.xnote{color:var(--muted);font-size:12px}
""".replace("@M@", MEAN_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--eval-dir", type=Path, default=None,
                    help="evaluate.py cache (default: artifacts/eval_cache/<run name> next to this script)")
    args = ap.parse_args()
    run_dir = args.run_dir
    out = args.output or run_dir / "report.html"

    groups = []
    for sub, label in GROUPS:
        runs = load_group(run_dir / sub)
        if not runs:
            print(f"skip {sub}: no runs with test rows", file=sys.stderr)
            continue
        mean = mean_curve(runs)
        peak = max(mean, key=lambda p: p[1])  # first max on ties
        total = TOTAL_FRAMES
        groups.append(dict(sub=sub, label=label, runs=runs, mean=mean, peak=peak, total=total))
    if not groups:
        sys.exit("no data found")

    max_frame = max(fr for g in groups for r in g["runs"].values() for fr in r)
    ymax = max(v[0] for g in groups for r in g["runs"].values() for v in r.values())
    ymax = ymax * 1.10  # axis limit = data max + 10%; ticks are drawn up to this

    body = []
    for g in groups:
        last = {n: max(r) for n, r in g["runs"].items()}
        unfinished = [n for n, f in last.items() if f < g["total"]]
        warn = ""
        if unfinished or len(g["runs"]) < EXPECTED_SEEDS:
            warn = (f'<p class="warn">Incomplete: {len(g["runs"])} runs found, latest frames '
                    + ", ".join(f"{n}: {fmt_frames(f)}" for n, f in last.items())
                    + f". Mean uses frames common to all seeds (up to {fmt_frames(g['mean'][-1][0])}).</p>")
        pf, pv = g["peak"]
        crash = sum(r[pf][1] for r in g["runs"].values()) / len(g["runs"])
        rows = "".join(f"<tr><td>{n}</td><td>{r[pf][0]:.2f}</td><td>{r[pf][1]:.2f}</td></tr>"
                       for n, r in g["runs"].items() if pf in r)
        legend = "".join(f'<span><i style="background:{SEED_COLORS[i % 6]}"></i>{n}</span>'
                         for i, n in enumerate(g["runs"]))
        legend += f'<span><i style="background:{MEAN_COLOR};height:4px"></i>mean over seeds</span>'
        body.append(
            f'<section class="card"><h2>{g["label"]}</h2>'
            f'<p class="sub">Peak of mean curve: <b>{pv:.2f}</b> at frame <b>{pf:,}</b> '
            f'(mean crash rate {crash:.2f}). For information only: robustness is evaluated at the final checkpoint of every run, not at this frame.</p>{warn}'
            + chart_svg(g["label"], g["runs"], g["mean"], g["peak"], max_frame, ymax)
            + f'<div class="legend">{legend}</div>'
            + f'<details><summary>Per-seed values at the peak frame</summary><table><tr><th>run</th><th>return</th><th>crash rate</th></tr>{rows}</table></details>'
            + "</section>")

    eval_dir = args.eval_dir or Path(__file__).resolve().parent / "artifacts" / "eval_cache" / run_dir.resolve().name
    ev = load_eval(eval_dir) if eval_dir.is_dir() else {}
    robust = robustness_section(ev, "final") + robustness_section(ev, "peak")
    if not robust:
        robust = '<p class="sub">No evaluation results found (run evaluate.py to add robustness sections).</p>'

    doc = (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>{html.escape(run_dir.name)} report</title><style>{CSS}</style></head><body><main>'
           f'<h1>{html.escape(run_dir.name)} &mdash; mini-test return</h1>'
           f'<p class="sub">Source: <code>train.log</code> (50 episodes, seeds 1000&ndash;1049, every 50k frames). '
           f'Thin lines = individual seeds; bright line = mean over seeds; dashed lines mark the mean curve&rsquo;s peak. '
           f'All charts share axes.</p>' + "".join(body) + robust + "</main></body></html>")
    out.write_text(doc)
    print(f"wrote {out}")
    for g in groups:
        print(f"{g['label']:12s} peak {g['peak'][1]:.2f} @ {g['peak'][0]:,} ({len(g['runs'])} runs)")


if __name__ == "__main__":
    main()
