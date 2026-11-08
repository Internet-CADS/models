#!/usr/bin/env python3
"""Model-free view of what an L-infinity observation perturbation means on highway-v0.

No checkpoints and no attacks: it records a short episode (ego keeps IDLE), converts the
normalised observation back to metres and m/s, and lets you pick any epsilon and see how far
each input is allowed to move (shaded box) plus random samples from that ball (dashed boxes).

Run from entries/04_wocar_dqn_r2/ with the repo venv:
    ../../.venv/bin/python visualize_perturbation.py -o perturbation.html [--seed 1] [--config config_highway_sadqn.json]
"""
import argparse
import json
import os
import sys
from pathlib import Path

ENTRY_DIR = Path(__file__).resolve().parent
os.chdir(ENTRY_DIR)
sys.path[:0] = [str(ENTRY_DIR)]

import numpy as np  # noqa: E402
from highway_adapter import LegacyHighwayEnv  # noqa: E402

EPS_CHOICES = [0.0005, 0.001, 0.0025, 0.004, 0.005, 0.007, 0.01, 0.02, 0.05]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", type=Path, default=Path("perturbation.html"))
    ap.add_argument("--config", default="config_highway_sadqn.json", help="env settings + training epsilon are read from here")
    ap.add_argument("--seed", type=int, default=1, help="episode seed for the recorded traffic")
    ap.add_argument("--train-eps", type=float, default=None, help="default: training epsilon from the config")
    ap.add_argument("--action", type=int, default=1, help="ego action used while recording (1 = IDLE)")
    args = ap.parse_args()

    cfg = json.load(open(ENTRY_DIR / args.config))
    train_eps = args.train_eps if args.train_eps is not None else float(cfg["training_config"]["epsilon"])
    env = LegacyHighwayEnv(cfg["env_id"], cfg["env_config"])
    env.seed(args.seed)
    s, done = env.reset(), False
    obs_list = []
    while not done:
        obs_list.append([float(v) for v in s])
        s, _, done, _ = env.step(args.action)
    fr = env.env.unwrapped.observation_type.features_range
    half = {k: (v[1] - v[0]) / 2 for k, v in fr.items()}  # metres (or m/s) per observation unit

    data = dict(obs=obs_list, half=half, lane_w=4.0, train_eps=train_eps, eps_choices=sorted(set(EPS_CHOICES + [train_eps])),
                note=f"episode seed {args.seed}, ego action {args.action}")
    args.output.write_text(TEMPLATE.replace("__DATA__", json.dumps(data)))
    print(f"wrote {args.output} ({len(obs_list)} steps; 1 obs unit = {half['x']:.0f} m x, {half['y']:.0f} m y, {half['vx']:.0f} m/s v; train eps {train_eps})")


TEMPLATE = r"""<title>Perturbation View</title>
<style>
:root{--bg:#fff;--fg:#1f2328;--muted:#6b7280;--bd:#e5e7eb;--card:#fff;--lane:#c5cad3;--car:#4e79a7;--ego:#e15759;--adv:#f28e2b}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e6e8eb;--muted:#9aa1ab;--bd:#2c3037;--card:#1b1e23;--lane:#4a505b;--car:#6c9bd0;--ego:#ff7b7d;--adv:#ffb05c}}
body{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
main{max-width:980px;margin:auto}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:0 0 6px}
.sub{color:var(--muted);margin:0 0 14px}.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:16px;margin:0 0 18px}
.ctl{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;margin-bottom:10px}select,input,button{font:inherit}
svg{width:100%;height:auto;display:block}table{border-collapse:collapse;font-size:13px}td,th{padding:4px 14px 4px 0;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:500}
.scroll{overflow-x:auto}.leg span{margin-right:16px;color:var(--muted);font-size:12px}.leg i{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:5px;vertical-align:-2px}
.big{font-size:15px}tr.cur td{font-weight:600}
</style>
<main>
<h1>What an observation perturbation looks like</h1>
<p class="sub" id="meta"></p>
<div class="card">
  <div class="ctl">
    <label>&epsilon; <select id="eps"></select></label>
    <label>Sampling <select id="mode"><option value="uniform">uniform in the ball</option><option value="corner">corner (every input &plusmn;&epsilon;)</option></select></label>
    <button id="resample">New sample</button>
    <label>Time step <input id="step" type="range" min="0" value="0"> <b id="stepv"></b></label>
    <label>Magnify &times;<input id="mag" type="range" min="1" max="100" value="1"> <b id="magv"></b></label>
  </div>
  <p class="big" id="headline"></p>
  <svg id="scene" viewBox="0 0 900 250" role="img" aria-label="top-down view of the observed vehicles"></svg>
  <div class="leg"><span><i style="background:var(--ego)"></i>ego (true)</span><span><i style="background:var(--car)"></i>other vehicle (true)</span><span><i style="background:var(--adv);opacity:.35"></i>region each vehicle may be shifted within</span><span><i style="background:none;border:2px dashed var(--adv)"></i>one perturbed sample</span></div>
</div>
<div class="card"><h2>In real-world terms: sensor error</h2>
  <p class="sub">Think of the observation as what the car&rsquo;s sensors report: own speed and lane position, and the measured gap, sideways offset and relative speed of each nearby vehicle. A perturbation means those readings are slightly wrong. Ordinary sensor noise is random; an <b>adversarial</b> perturbation is the worst combination of errors within the same limits, chosen to confuse this particular network. The dashed &ldquo;perceived&rdquo; values below are one random sample from the allowed range.</p>
  <div class="scroll" id="rw"></div>
  <p class="sub" style="margin-top:10px">The ego&rsquo;s absolute position along the road is also an input and also perturbed (&plusmn;<span id="egox"></span> m), but it is not shown as a movement because it has no effect on the scene around the car.</p>
</div>
<div class="card"><h2>Allowed shift per input, in physical units</h2><p class="sub">Every one of the 50 inputs may move independently by up to &plusmn;&epsilon; in normalised units. The table converts that to metres and m/s and compares it with something familiar.</p><div class="scroll" id="tbl"></div></div>
</main>
<script>
const D=__DATA__, $=id=>document.getElementById(id), NS="http://www.w3.org/2000/svg";
const F=["presence","x","y","vx","vy"], H=D.half, CAR_L=4.5, CAR_W=2.0;
let noise=[]; // 50 values in [-1,1] (unit noise), scaled by eps at draw time
function resample(){const corner=$("mode").value==="corner"; noise=Array.from({length:50},()=>corner?(Math.random()<.5?-1:1):Math.random()*2-1);}
$("meta").textContent=`Traffic: ${D.note}. Normalisation: 1 observation unit = ${H.x} m (x), ${H.y} m (y), ${H.vx} m/s (vx, vy). No model is involved; this is only the geometry of the allowed perturbation.`;
D.eps_choices.forEach(e=>$("eps").add(new Option(e+(e===D.train_eps?" (training)":""),e))); $("eps").value=D.train_eps;
$("step").max=D.obs.length-1;
const el=(n,a={},p)=>{const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e};
const fmt=(v,d=2)=>v.toFixed(d);
function render(){
  const eps=+$("eps").value, mag=+$("mag").value, t=+$("step").value; $("magv").textContent=mag; $("stepv").textContent=t;
  const o=D.obs[t], at=(r,f)=>o[r*5+F.indexOf(f)]*H[f==="x"?"x":f==="y"?"y":"vx"];
  const dx=eps*H.x, dy=eps*H.y, dv=eps*H.vx;
  $("headline").innerHTML=`At &epsilon; = <b>${eps}</b> each vehicle's position can be off by up to <b>${fmt(dx)} m</b> lengthwise and <b>${fmt(dy,3)} m</b> sideways, and its speed by <b>${fmt(dv)} m/s</b>.`;
  const svg=$("scene"); svg.innerHTML="";
  const rows=[...Array(10).keys()].filter(r=>o[r*5]>0.5);
  const xs=rows.map(r=>r===0?0:at(r,"x")), xmin=Math.min(-20,...xs)-8, xmax=Math.max(60,...xs)+8, W=900;
  const sx=x=>(x-xmin)/(xmax-xmin)*W, pm=W/(xmax-xmin), sy=y=>40+(y+2)/16*170;
  for(let i=0;i<=4;i++) el("line",{x1:0,x2:W,y1:sy(i*D.lane_w-2),y2:sy(i*D.lane_w-2),stroke:"var(--lane)","stroke-width":i%4?1:2,"stroke-dasharray":i%4?"10 8":""},svg);
  const egoY=at(0,"y"), ym=170/16;
  rows.forEach(r=>{
    const ax=r===0?0:at(r,"x"), ay=r===0?egoY:egoY+at(r,"y");
    const n=(f)=>noise[r*5+F.indexOf(f)]*eps*H[f==="x"?"x":f==="y"?"y":"vx"]*mag;
    // ego x is an absolute road coordinate (carries no meaning in this scene): perturbed in the input but not drawn as a shift
    const bx=ax+(r===0?0:n("x")), by=ay+n("y"), rx=r===0?0:dx;
    el("rect",{x:sx(ax)-CAR_L*pm/2-rx*mag*pm,y:sy(ay)-CAR_W*ym/2-dy*mag*ym,width:CAR_L*pm+2*rx*mag*pm,height:CAR_W*ym+2*dy*mag*ym,fill:"var(--adv)",opacity:.28},svg);
    el("rect",{x:sx(ax)-CAR_L*pm/2,y:sy(ay)-CAR_W*ym/2,width:CAR_L*pm,height:CAR_W*ym,rx:2,fill:r===0?"var(--ego)":"var(--car)"},svg);
    el("rect",{x:sx(bx)-CAR_L*pm/2,y:sy(by)-CAR_W*ym/2,width:CAR_L*pm,height:CAR_W*ym,rx:2,fill:"none",stroke:"var(--adv)","stroke-width":2,"stroke-dasharray":"5 3"},svg);
    const tx=el("text",{x:sx(ax),y:sy(ay)+4,"text-anchor":"middle","font-size":11,fill:"#fff"},svg); tx.textContent=r===0?"ego":r;
  });
  const sc=el("text",{x:8,y:242,fill:"var(--muted)","font-size":11},svg);
  sc.textContent=mag===1?"drawn to scale (vehicles are 4.5 m × 2 m, lanes 4 m wide)":`allowed shift and sample displacement magnified ×${mag}; vehicles and lanes to scale`;
  $("egox").textContent=fmt(dx);
  // real-world readout: what a driver-level sensor would report, true vs perceived (this sample)
  const nz=(r,f)=>noise[r*5+F.indexOf(f)]*eps*H[f==="x"?"x":f==="y"?"y":"vx"];
  const ahead=rows.filter(r=>r>0&&at(r,"x")>0).sort((a,b)=>(Math.abs(at(a,"y"))>2)-(Math.abs(at(b,"y"))>2)||at(a,"x")-at(b,"x"))[0];
  const line=(what,unit,tv,pv,maxe,d=2)=>`<tr><td>${what}</td><td>${fmt(tv,d)} ${unit}</td><td>${fmt(pv,d)} ${unit}</td><td>${fmt(pv-tv,d)}</td><td>&plusmn;${fmt(maxe,d)}</td></tr>`;
  let rw=`<table><tr><th>reading</th><th>true</th><th>agent perceives (this sample)</th><th>error</th><th>max error</th></tr>`;
  rw+=line("Own speed","m/s",at(0,"vx"),at(0,"vx")+nz(0,"vx"),dv);
  rw+=line("Own lateral position","m",egoY,egoY+nz(0,"y"),dy,3);
  if(ahead!==undefined){
    rw+=line(`Gap to car ahead (vehicle ${ahead})`,"m",at(ahead,"x"),at(ahead,"x")+nz(ahead,"x"),dx);
    rw+=line(`&nbsp;&nbsp;its sideways offset`,"m",at(ahead,"y"),at(ahead,"y")+nz(ahead,"y"),dy,3);
    rw+=line(`&nbsp;&nbsp;its speed relative to you`,"m/s",at(ahead,"vx"),at(ahead,"vx")+nz(ahead,"vx"),dv);
  } else rw+="<tr><td colspan='5'>No car ahead in this frame.</td></tr>";
  $("rw").innerHTML=rw+"</table>";
  // table
  let h="<table><tr><th>&epsilon;</th><th>x (m)</th><th>y (m)</th><th>vx, vy (m/s)</th><th>x vs car length</th><th>y vs lane width</th><th>speed vs 25 m/s</th></tr>";
  D.eps_choices.forEach(e=>{h+=`<tr class="${e===eps?"cur":""}"><td>${e}${e===D.train_eps?" (train)":""}</td><td>&plusmn;${fmt(e*H.x)}</td><td>&plusmn;${fmt(e*H.y,3)}</td><td>&plusmn;${fmt(e*H.vx)}</td><td>${fmt(e*H.x/CAR_L*100,0)}%</td><td>${fmt(e*H.y/D.lane_w*100,1)}%</td><td>${fmt(e*H.vx/25*100,1)}%</td></tr>`});
  $("tbl").innerHTML=h+"</table><p class='sub'>The presence flag (0/1) is perturbed by the same &plusmn;&epsilon; but has no physical unit, so it is not drawn.</p>";
}
["eps","step","mag"].forEach(i=>$(i).addEventListener("input",render));
$("mode").addEventListener("input",()=>{resample();render()}); $("resample").addEventListener("click",()=>{resample();render()});
resample(); render();
</script>
"""

if __name__ == "__main__":
    main()
