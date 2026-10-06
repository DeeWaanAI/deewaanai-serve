#!/usr/bin/env python3
"""make_crossover_figs.py — SVG figures from frozen crossover + union results."""
import csv, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/"results/crossover/figures"; FIG.mkdir(parents=True,exist_ok=True)

# ---- Figure A: p05 vs concurrency, pinned vs llama ----
rows=list(csv.DictReader(open(ROOT/"results/crossover/p05/p05_levels.csv")))
def series(arm):
    d={}
    for r in rows:
        if r["arm"]==arm: d[int(r["level"])]=float(r["p05_tok_s"])
    return sorted(d.items())
llama=series("llama_xover"); pinned=series("pinned_xover")
W,H,L,R,T,B=760,470,70,30,40,60
xs=[l for l,_ in pinned]; xall=sorted(set(xs)|set(l for l,_ in llama))
ymax=70.0; xmax=32
X=lambda v:L+(v/xmax)*(W-L-R+L)  # placeholder
def px(v): return L+(v-1)/(xmax-1)*(W-L-R)
def py(v): return T+(1-min(v,ymax)/ymax)*(H-T-B)
def path(s): return " ".join(("M" if i==0 else "L")+f"{px(l):.1f},{py(v):.1f}" for i,(l,v) in enumerate(s))
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Helvetica,Arial,sans-serif">',
     f'<rect width="{W}" height="{H}" fill="#fff"/>',
     f'<text x="{W/2}" y="24" text-anchor="middle" font-size="17" font-weight="700" fill="#1a1a1a">Figure A — Concurrent users vs p05 per-user throughput (OLMoE, 16GB M1 Pro)</text>']
# axes
svg.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H-B}" stroke="#333"/>')
svg.append(f'<line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" stroke="#333"/>')
for gv in [0,10,20,30,40,50,60,70]:
    y=py(gv); svg.append(f'<line x1="{L}" y1="{y:.0f}" x2="{W-R}" y2="{y:.0f}" stroke="#eee"/>')
    svg.append(f'<text x="{L-8}" y="{y+4:.0f}" text-anchor="end" font-size="11" fill="#666">{gv}</text>')
for lv in xall:
    x=px(lv); svg.append(f'<text x="{x:.0f}" y="{H-B+18}" text-anchor="middle" font-size="11" fill="#666">{lv}</text>')
svg.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-14}" text-anchor="middle" font-size="12" fill="#333">concurrent users</text>')
svg.append(f'<text x="18" y="{(T+H-B)/2:.0f}" font-size="12" fill="#333" transform="rotate(-90 18 {(T+H-B)/2:.0f})">p05 tok/s per user</text>')
# SLO floor
y=py(10); svg.append(f'<line x1="{L}" y1="{y:.0f}" x2="{W-R}" y2="{y:.0f}" stroke="#c0392b" stroke-dasharray="6 4"/>')
svg.append(f'<text x="{W-R}" y="{y-6:.0f}" text-anchor="end" font-size="11" fill="#c0392b">SLO floor = 10 tok/s</text>')
# ceiling markers
for lv,col,lab in [(8,"#d62728","llama ceiling 8"),(16,"#2f7a2f","pinned ceiling 16")]:
    svg.append(f'<line x1="{px(lv):.0f}" y1="{T}" x2="{px(lv):.0f}" y2="{H-B}" stroke="{col}" stroke-width="1" stroke-dasharray="3 3" opacity="0.6"/>')
# lines
svg.append(f'<path d="{path(llama)}" fill="none" stroke="#d62728" stroke-width="2.5"/>')
svg.append(f'<path d="{path(pinned)}" fill="none" stroke="#2f7a2f" stroke-width="2.5"/>')
for s,col in [(llama,"#d62728"),(pinned,"#2f7a2f")]:
    for l,v in s: svg.append(f'<circle cx="{px(l):.0f}" cy="{py(v):.0f}" r="3.5" fill="{col}"/>')
# legend
svg.append(f'<rect x="{L+14}" y="{T+8}" width="210" height="46" fill="#fff" stroke="#ddd"/>')
svg.append(f'<line x1="{L+22}" y1="{T+24}" x2="{L+50}" y2="{T+24}" stroke="#2f7a2f" stroke-width="3"/><text x="{L+56}" y="{T+28}" font-size="12" fill="#1a1a1a">pinned (ours) — ceiling 16</text>')
svg.append(f'<line x1="{L+22}" y1="{T+42}" x2="{L+50}" y2="{T+42}" stroke="#d62728" stroke-width="3"/><text x="{L+56}" y="{T+46}" font-size="12" fill="#1a1a1a">llama.cpp — ceiling 8</text>')
svg.append('</svg>')
(FIG/"figA_p05_curves.svg").write_text("\n".join(svg)); print("wrote figA_p05_curves.svg")

# ---- Figure B: resident experts vs concurrency (union) ----
u=json.load(open(ROOT/"results/telemetry/olmoe_trace400_union.json"))
c95=u["curve"]["0.95"]; c90=u["curve"]["0.9"]; pool=u["experts_per_layer"]
ks=sorted(int(k) for k in c95)
def pb(v): return L+(v-1)/(64-1)*(W-L-R)
def qb(v): return T+(1-min(v,pool)/pool)*(H-T-B)
def pbpath(c): return " ".join(("M" if i==0 else "L")+f"{pb(k):.1f},{qb(c[str(k)]):.1f}" for i,k in enumerate(ks))
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Helvetica,Arial,sans-serif">',
     f'<rect width="{W}" height="{H}" fill="#fff"/>',
     f'<text x="{W/2}" y="24" text-anchor="middle" font-size="17" font-weight="700" fill="#1a1a1a">Figure B — Resident experts needed vs concurrency (OLMoE, 64/layer)</text>',
     f'<text x="{W/2}" y="40" text-anchor="middle" font-size="11.5" fill="#666">sub-linear growth = the concurrency amortization behind the pinned win</text>']
svg.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H-B}" stroke="#333"/><line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" stroke="#333"/>')
for gv in [0,16,32,48,64]:
    y=qb(gv); svg.append(f'<line x1="{L}" y1="{y:.0f}" x2="{W-R}" y2="{y:.0f}" stroke="#eee"/><text x="{L-8}" y="{y+4:.0f}" text-anchor="end" font-size="11" fill="#666">{gv}</text>')
for kv in [1,2,4,8,16,32,64]:
    x=pb(kv); svg.append(f'<text x="{x:.0f}" y="{H-B+18}" text-anchor="middle" font-size="11" fill="#666">{kv}</text>')
yp=qb(pool); svg.append(f'<line x1="{L}" y1="{yp:.0f}" x2="{W-R}" y2="{yp:.0f}" stroke="#999" stroke-dasharray="5 4"/><text x="{W-R}" y="{yp-5:.0f}" text-anchor="end" font-size="11" fill="#777">full pool = 64</text>')
svg.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-14}" text-anchor="middle" font-size="12" fill="#333">concurrent users (k)</text>')
svg.append(f'<text x="18" y="{(T+H-B)/2:.0f}" font-size="12" fill="#333" transform="rotate(-90 18 {(T+H-B)/2:.0f})">experts resident per layer</text>')
svg.append(f'<path d="{pbpath(c95)}" fill="none" stroke="#1f77b4" stroke-width="2.5"/>')
svg.append(f'<path d="{pbpath(c90)}" fill="none" stroke="#ff7f0e" stroke-width="2.5"/>')
for k in ks:
    svg.append(f'<circle cx="{pb(k):.0f}" cy="{qb(c95[str(k)]):.0f}" r="3" fill="#1f77b4"/><circle cx="{pb(k):.0f}" cy="{qb(c90[str(k)]):.0f}" r="3" fill="#ff7f0e"/>')
svg.append(f'<rect x="{L+14}" y="{T+8}" width="230" height="46" fill="#fff" stroke="#ddd"/>')
svg.append(f'<line x1="{L+22}" y1="{T+24}" x2="{L+50}" y2="{T+24}" stroke="#1f77b4" stroke-width="3"/><text x="{L+56}" y="{T+28}" font-size="12">cover 95% of traffic (36→57)</text>')
svg.append(f'<line x1="{L+22}" y1="{T+42}" x2="{L+50}" y2="{T+42}" stroke="#ff7f0e" stroke-width="3"/><text x="{L+56}" y="{T+46}" font-size="12">cover 90% of traffic (31→51)</text>')
svg.append('</svg>')
(FIG/"figB_resident_growth.svg").write_text("\n".join(svg)); print("wrote figB_resident_growth.svg")
