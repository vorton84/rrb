

import os
import re
import sys
from collections import defaultdict

BASE = os.environ.get("EVAL_LOG_DIR", os.getcwd())

RE_ACC = re.compile(r"val-core/math_dapo/acc/mean@128:([0-9.]+)")
RE_CKPT = re.compile(r"resume_from_path=\S*?/([^/\s]+)/global_step_(\d+)")

def scan():
    out = defaultdict(lambda: defaultdict(dict))
    for fn in sorted(os.listdir(BASE)):
        if not fn.endswith("_n4.log"):
            continue
        if fn.startswith("eval25_"):
            bench = "25"
        elif fn.startswith("eval26_"):
            bench = "26"
        elif fn.startswith("eval_"):
            bench = "24"
        else:
            continue
        path = os.path.join(BASE, fn)
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError:
            continue
        ck = RE_CKPT.search(text)
        acc = RE_ACC.findall(text)
        if not ck or not acc:
            continue
        exp, step = ck.group(1), int(ck.group(2))
        out[exp][step][bench] = float(acc[-1]) * 100.0
    return out

def stats(vals):
    n = len(vals)
    if n == 0:
        return None
    m = sum(vals) / n
    if n == 1:
        return m, None, 1
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, (var ** 0.5) / (n ** 0.5), n

def fmt(st):
    if st is None:
        return "–"
    m, sem, n = st
    return f"{m:.2f} ± {sem:.2f} (n={n})" if sem is not None else f"{m:.2f} (n=1)"

def main():
    args = [a for a in sys.argv[1:]]
    as_md = "--md" in args
    args = [a for a in args if a != "--md"]
    filt = args[0] if args else ""

    data = scan()
    if not data:
        print(f"no mean@128 logs found under {BASE}")
        return 1

    for exp in sorted(data):
        if filt and filt not in exp:
            continue
        steps = sorted(data[exp])
        v24 = [data[exp][s]["24"] for s in steps if "24" in data[exp][s]]
        v25 = [data[exp][s]["25"] for s in steps if "25" in data[exp][s]]
        pooled_steps = [s for s in steps if "24" in data[exp][s] and "25" in data[exp][s]]
        vp = [(data[exp][s]["24"] + data[exp][s]["25"]) / 2 for s in pooled_steps]
        v26 = [data[exp][s]["26"] for s in steps if "26" in data[exp][s]]

        if as_md:
            print(f"\n**{exp}**\n")
            print("| step | AIME 2024 | AIME 2025 | pooled 60 | AIME 2026 |")
            print("|---|---|---|---|---|")
            for s in steps:
                d = data[exp][s]
                a = f"{d['24']:.2f}" if "24" in d else "–"
                b = f"{d['25']:.2f}" if "25" in d else "–"
                p = f"{(d['24'] + d['25']) / 2:.2f}" if "24" in d and "25" in d else "–"
                c = f"{d['26']:.2f}" if "26" in d else "–"
                print(f"| {s} | {a} | {b} | {p} | {c} |")
            print(f"| **mean** | **{fmt(stats(v24))}** | **{fmt(stats(v25))}** | **{fmt(stats(vp))}** | **{fmt(stats(v26))}** |")
        else:
            print(f"\n=== {exp} ===")
            print(f"{'step':<8}{'AIME24':>10}{'AIME25':>10}{'pooled60':>12}{'AIME26':>10}")
            print(f"{'----':<8}{'------':>10}{'------':>10}{'--------':>12}{'------':>10}")
            for s in steps:
                d = data[exp][s]
                a = f"{d['24']:.2f}" if "24" in d else "–"
                b = f"{d['25']:.2f}" if "25" in d else "–"
                p = f"{(d['24'] + d['25']) / 2:.2f}" if "24" in d and "25" in d else "–"
                c = f"{d['26']:.2f}" if "26" in d else "–"
                print(f"{s:<8}{a:>10}{b:>10}{p:>12}{c:>10}")
            print(f"{'MEAN24':<8}{fmt(stats(v24))}")
            print(f"{'MEAN25':<8}{fmt(stats(v25))}")
            print(f"{'POOLED':<8}{fmt(stats(vp))}")
            print(f"{'MEAN26':<8}{fmt(stats(v26))}")

    print("\nnote: pooled = (AIME24 + AIME25)/2, only where both were measured; AIME 2026 is a separate column.")
    print("      SEM is across checkpoints; add ~0.7pp per-checkpoint sampling noise")
    print("      (~0.5pp pooled) before comparing two runs.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
