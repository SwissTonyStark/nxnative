#!/usr/bin/env python3
# pace40.py <log>: per 300-frame window the frame time / GPU time percentiles, and per profiled frame ([gpuprof]) the 3D resolution the game chose
# (the largest depth pass with many draws) and the frame's GPU sum. Also the pacing / frame-step lines.
import re, sys
log = sys.argv[1]
fr = {}
for l in open(log, errors='replace'):
    if re.search(r'40 fps|present tick|pacing mode|counter timestamps at|frame step cap|frame step: hold|display link: frame latency|the screen runs at|3D views rendered', l):
        print(l.strip()[:200])
    m = re.search(r'timing up to frame (\d+): frame time p50 ([0-9.]+) / p95 ([0-9.]+) / max ([0-9.]+).*GPU p50 ([0-9.]+) / p95 ([0-9.]+) / max ([0-9.]+)', l)
    if m:
        f, a, b, c, g, g95, gm = m.groups()
        print(f'F{f:>6} frame p50 {a} p95 {b} max {c} | GPU p50 {g} p95 {g95} max {gm}')
    m = re.search(r'\[gpuprof\] F(\d+) +(\d+) +([0-9.]+) us (.*)', l)
    if m: fr.setdefault(int(m.group(1)), []).append((float(m.group(3)), m.group(4)))
for f, v in sorted(fr.items()):
    tot = sum(x[0] for x in v) / 1000
    big = [(int(re.search(r'draws (\d+)', d).group(1)), re.search(r'(\d+x\d+)', d).group(1), us / 1000) for us, d in v if 'render pass' in d and 'depth 1' in d and re.search(r'draws (\d+)', d)]
    big.sort(reverse=True)
    main = big[0] if big else None
    print(f'  profiled F{f}: GPU sum {tot:.1f} ms; main pass {main[1] if main else "-"} ({main[0] if main else 0} draws, {main[2] if main else 0:.1f} ms); encoders {len(v)}')
