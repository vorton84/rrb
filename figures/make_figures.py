
import json, os, statistics as S
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures"); os.makedirs(OUT, exist_ok=True)
OBS = os.path.join(HERE, "obs.json")

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "figure.dpi": 150})
C = {"dapo": "#222222", "g128": "#1f77b4", "g96": "#d62728", "b15": "#2ca02c", "gen7": "#9467bd",
     "band": "#bbbbbb", "hi": "#d62728", "grey": "#888888"}

AUDIT = [
 ("DAPO seed 1",            33.48, 18.91, 26.20, "ref"),
 ("FIPO",                   33.09, 16.46, 24.77, "imp"),
 ("FIPO-K",                 34.51, 14.16, 24.33, "imp"),
 ("PSO v1",                 30.49, 14.37, 22.43, "adv"),
 ("PSO v5 A=1.2",           31.39, 17.20, 24.29, "adv"),
 ("PSO v5 A=1.5",           33.32, 14.77, 24.05, "adv"),
 ("PSO v5 A=2.0",           32.69, 13.34, 23.02, "adv"),
 ("DAPO+KTSC",              31.97, 16.75, 24.36, "step"),
 ("PSO A=1.5+KTSC",         30.44, 14.61, 22.53, "step"),
 ("PSO A=2.0+KTSC",         34.94, 12.42, 23.68, "step"),
 ("Dr.GRPO",                32.66, 16.01, 24.33, "step"),
 ("IRT sel., gen 256",      31.74, 15.14, 23.44, "prompt"),
 ("IRT sel., gen 128",      32.57, 18.63, 25.60, "prompt"),
 ("random, gen 128 s1",     34.34, 15.36, 24.85, "prompt"),
]
DAPO_MEAN, DAPO_SD = 25.31, 0.99

def fig1_seed_band():
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    names = [a[0] for a in AUDIT]; pooled = [a[3] for a in AUDIT]
    y = list(range(len(AUDIT)))[::-1]
    ax.axvspan(DAPO_MEAN - 2*DAPO_SD, DAPO_MEAN + 2*DAPO_SD, color=C["band"], alpha=0.35, lw=0, label="DAPO mean $\\pm$ 2 SD (3 seeds)")
    ax.axvspan(DAPO_MEAN - DAPO_SD, DAPO_MEAN + DAPO_SD, color=C["band"], alpha=0.6, lw=0, label="$\\pm$ 1 SD")
    ax.axvline(DAPO_MEAN, color=C["dapo"], lw=1)
    for yi, p, n in zip(y, pooled, names):
        out = abs(p - DAPO_MEAN) > 2*DAPO_SD
        ax.plot(p, yi, "o", ms=5, color=C["hi"] if out else C["dapo"], zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlabel("pooled AIME 2024+2025, mean@128 at steps 320/350/380 (%)")
    ax.set_xlim(21.5, 27.2)
    ax.legend(loc="upper left", fontsize=7.5, frameon=True, framealpha=0.95, edgecolor="none")
    ax.grid(axis="x", lw=0.3, alpha=0.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_seed_band.pdf")); plt.close(fig)

def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: -xs[i]); r = [0]*len(xs)
    for k, i in enumerate(order): r[i] = k + 1
    return r

def fig2_rank_inversion():
    runs = [a for a in AUDIT if a[0] != "FIPO-K"]
    r24, r25 = rank([a[1] for a in runs]), rank([a[2] for a in runs])
    n = len(runs)
    rho = 1 - 6*sum((a-b)**2 for a, b in zip(r24, r25)) / (n*(n*n-1))
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot([0.5, n+0.5], [0.5, n+0.5], "--", color=C["grey"], lw=0.8)
    for (name, *_), a, b in zip(runs, r24, r25):
        ax.plot(a, b, "o", color=C["dapo"], ms=4)
        ax.annotate(name, (a, b), fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("rank on AIME 2024 (1 = best)"); ax.set_ylabel("rank on AIME 2025 (1 = best)")
    ax.set_xlim(0.5, n+0.5); ax.set_ylim(0.5, n+0.5); ax.invert_xaxis(); ax.invert_yaxis()
    ax.set_title(f"Spearman $\\rho$ = {rho:+.2f}, {n} single-seed runs", fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_rank_inversion.pdf")); plt.close(fig)

def fig3_mixed_rate():
    obs = json.load(open(OBS))
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    def line(key, color, label, ls="-", lw=1.3, alpha=1.0):
        pts = obs[key]; ax.plot([g for g, _ in pts], [r for _, r in pts], ls, color=color, lw=lw, alpha=alpha, label=label)
    for i, k in enumerate(("math7b_g128_s1", "math7b_g128_s2", "math7b_g128_s3")):
        line(k, C["g128"], "Qwen2.5-Math-7B, gen 128 (3 seeds)" if i == 0 else None, alpha=0.8, lw=1.0)
    line("math7b_g96", C["g96"], "Qwen2.5-Math-7B, gen 96")
    line("gen7b_g256", C["gen7"], "Qwen2.5-7B (general), gen 256")
    line("math15b_g256", C["b15"], "Qwen2.5-Math-1.5B, gen 256")
    ax.axhline(25, color=C["dapo"], lw=0.9, ls=":", label="25%: the 'discard statistic' (64/256)")
    ax.axhline(50, color=C["g128"], lw=0.9, ls=":"); ax.axhline(100*64/96, color=C["g96"], lw=0.9, ls=":")
    ax.text(392, 51, "64/128", fontsize=7, color=C["g128"], ha="right", va="bottom")
    ax.text(392, 100*64/96+1, "64/96", fontsize=7, color=C["g96"], ha="right", va="bottom")
    ax.set_xlabel("training step"); ax.set_ylabel("mixed-outcome groups (% of generated)")
    ax.set_xlim(0, 400); ax.set_ylim(0, 100)
    ax.legend(fontsize=7, frameon=True, framealpha=0.95, edgecolor="none", loc="lower right")
    ax.grid(lw=0.3, alpha=0.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_mixed_rate.pdf")); plt.close(fig)

def fig4_gen_batch():

    gb = [96, 128, 256]; roll = [96*1.261, 128*1.000, 256*1.000]; frac2 = [26.1, 0.0, 0.0]
    fig, ax = plt.subplots(figsize=(3.8, 2.9))
    bars = ax.bar([str(g) for g in gb], roll, color=[C["g96"], C["g128"], C["dapo"]], width=0.6)
    for b, r, f in zip(bars, roll, frac2):
        ax.text(b.get_x()+b.get_width()/2, r+5, f"{r:.0f}/step", ha="center", fontsize=8)
        ax.text(b.get_x()+b.get_width()/2, 12, f"2nd batch\n{f:.0f}% of steps", ha="center", fontsize=7, color="white")
    ax.axhline(64/0.70, color=C["grey"], ls="--", lw=0.9)
    ax.text(1.0, 64/0.70+4, "64 / 0.70 = 91 prompts (predicted minimum)", fontsize=7, ha="center", color=C["grey"])
    ax.set_xlabel("generation batch (prompts)"); ax.set_ylabel("prompts generated per step")
    ax.set_ylim(0, 300)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_gen_batch.pdf")); plt.close(fig)

def fig5_dose_response():

    stdw = [0.0, 0.048, 0.119, 0.239]; a25 = [18.91, 17.20, 14.77, 13.34]; a24 = [33.48, 31.39, 33.32, 32.69]
    ktsc = [(0.119, 14.61), (0.239, 12.42)]
    SD25, SD24 = 1.16, 1.68
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.7), sharex=True)
    for ax, y, sd, t in ((axs[0], a25, SD25, "AIME 2025"), (axs[1], a24, SD24, "AIME 2024")):
        ax.axhspan(y[0]-sd, y[0]+sd, color=C["band"], alpha=0.5, lw=0, label="DAPO $\\pm$ 1 seed SD")
        ax.plot(stdw, y, "o-", color=C["dapo"], label="PSO v5, uncontrolled")
        if t == "AIME 2025":
            ax.plot([k[0] for k in ktsc], [k[1] for k in ktsc], "s", color=C["hi"], label="PSO v5 + KL-targeted step")
        else:
            ax.plot([0.119, 0.239], [30.44, 34.94], "s", color=C["hi"])
        ax.set_title(t, fontsize=9); ax.set_xlabel("std of advantage weights, std($w$)")
        ax.grid(lw=0.3, alpha=0.5)
    axs[0].set_ylabel("mean@128 accuracy (%)"); axs[0].legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_dose_response.pdf")); plt.close(fig)

if __name__ == "__main__":
    fig1_seed_band(); fig2_rank_inversion(); fig3_mixed_rate(); fig4_gen_batch(); fig5_dose_response()
    print("wrote:", sorted(os.listdir(OUT)))
