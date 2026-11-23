#!/usr/bin/env python3
"""perfchart writes docs/perf.svg, the README's chart of goaec3's frame time
as a share of libwebrtc's, from the ratios in GROUPS (bench.yml rows, Go/C++).

Usage: python3 scripts/perfchart.py docs/perf.svg
"""
import sys

GROUPS = [
    ("Apple M4 Max · NEON", [0.65, 0.65, 0.72]),
    ("Neoverse-N2 · NEON", [0.80, 0.80, 0.81]),
    ("AMD EPYC Zen 4 · AVX2", [0.75, 0.74, 0.71]),
    ("AMD EPYC Zen 3 · SSE2", [0.73, 0.73, 0.79]),
]
SERIES = ["16 kHz mono", "48 kHz mono", "48 kHz stereo"]


def main(out):
    W, left, right, top, bh, gap, ggap = 720, 190, 60, 78, 14, 2, 22
    plotw = W - left - right
    x = lambda v: left + plotw * v
    gh = 3 * bh + 2 * gap
    H = top + len(GROUPS) * gh + (len(GROUPS) - 1) * ggap + 48
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif" '
         'role="img" aria-label="goaec3 frame time as a percentage of libwebrtc">',
         '<style>'
         '.bg{fill:#fcfcfb}.t1{fill:#0b0b0b}.t2{fill:#52514e}.t3{fill:#7a7975}.grid{stroke:#e4e3df}.ref{stroke:#0b0b0b}'
         '.s0{fill:#2a78d6}.s1{fill:#eb6834}.s2{fill:#1baf7a}'
         '@media (prefers-color-scheme: dark){.bg{fill:#1a1a19}.t1{fill:#f0efec}.t2{fill:#c3c2b7}.t3{fill:#9a998f}'
         '.grid{stroke:#33332f}.ref{stroke:#f0efec}.s0{fill:#3987e5}.s1{fill:#d95926}.s2{fill:#199e70}}'
         'text{font-size:12px}</style>',
         f'<rect class="bg" width="{W}" height="{H}" rx="8"/>',
         '<text class="t1" x="24" y="30" style="font-size:15px;font-weight:600">goaec3 time per 10 ms frame, % of libwebrtc m141</text>',
         '<text class="t2" x="24" y="49">GOEXPERIMENT=simd · same output bit for bit · shorter is faster</text>']
    lx = left
    for i, s in enumerate(SERIES):
        o.append(f'<rect class="s{i}" x="{lx}" y="58" width="10" height="10" rx="2"/>'
                 f'<text class="t2" x="{lx + 15}" y="67">{s}</text>')
        lx += 15 + len(s) * 6.6 + 22
    yb, ye = top, H - 36
    for v in (0, 0.25, 0.5, 0.75):
        o.append(f'<line class="grid" x1="{x(v):.1f}" y1="{yb - 4}" x2="{x(v):.1f}" y2="{ye}" stroke-width="1"/>')
        o.append(f'<text class="t3" x="{x(v):.1f}" y="{ye + 16}" text-anchor="middle">{int(v * 100)}%</text>')
    y = top
    for name, vals in GROUPS:
        o.append(f'<text class="t1" x="{left - 12}" y="{y + gh / 2 + 4:.1f}" text-anchor="end">{name}</text>')
        for i, v in enumerate(vals):
            w = x(v) - left
            pct = int(round(v * 100))
            o.append(f'<path class="s{i}" d="M{left},{y} h{w - 4:.1f} a4,4 0 0 1 4,4 v{bh - 8} a4,4 0 0 1 -4,4 '
                     f'h-{w - 4:.1f} z"><title>{name}, {SERIES[i]}: {pct}% of libwebrtc</title></path>')
            o.append(f'<text class="t2" x="{x(v) + 5:.1f}" y="{y + bh - 3}">{pct}%</text>')
            y += bh + gap
        y += ggap - gap
    o.append(f'<line class="ref" x1="{x(1):.1f}" y1="{yb - 4}" x2="{x(1):.1f}" y2="{ye}" stroke-width="2" stroke-dasharray="4 3"/>')
    o.append(f'<text class="t1" x="{x(1):.1f}" y="{ye + 16}" text-anchor="middle" style="font-weight:600">libwebrtc = 100%</text>')
    o.append('</svg>')
    with open(out, "w") as f:
        f.write("\n".join(o) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/perf.svg")
