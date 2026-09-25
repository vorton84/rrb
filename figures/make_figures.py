                      
\
\
\
\
\
\
\
   
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures"); os.makedirs(OUT, exist_ok=True)
OBS = os.path.join(HERE, "obs.json")

                                                                                     
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
    "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.grid": True, "grid.color": "#d9d9d9", "grid.linewidth": 0.4, "grid.alpha": 1.0, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
INK   = "#1f1f1f"
BLUE  = "#0072B2"
VERM  = "#D55E00"
GREEN = "#009E73"
PURP  = "#CC79A7"
ORNG  = "#E69F00"
BAND1 = "#c9d3de"           
BAND2 = "#e6ebf0"           
MUTED = "#7a7a7a"

def save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig)

                                                                              
AUDIT = [                                
 ("DAPO seed 1",            33.48, 18.91, 26.20),
 ("FIPO",                   33.09, 16.46, 24.77),
 ("FIPO-K",                 34.51, 14.16, 24.33),
 ("PSO v1",                 30.49, 14.37, 22.43),
 ("PSO v5, $A$=1.2",        31.39, 17.20, 24.29),
 ("PSO v5, $A$=1.5",        33.32, 14.77, 24.05),
 ("PSO v5, $A$=2.0",        32.69, 13.34, 23.02),
 ("DAPO + KTSC",            31.97, 16.75, 24.36),
 ("PSO $A$=1.5 + KTSC",     30.44, 14.61, 22.53),
 ("PSO $A$=2.0 + KTSC",     34.94, 12.42, 23.68),
 ("Dr. GRPO",               32.66, 16.01, 24.33),
 ("IRT selection, gen 256", 31.74, 15.14, 23.44),
 ("IRT selection, gen 128", 32.57, 18.63, 25.60),
 ("random, gen 128, seed 1",34.34, 15.36, 24.85),
]
DAPO_MEAN, DAPO_SD = 25.31, 0.99
PI_HALF = 2.40                                                                                          

def fig1_seed_band():
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    names = [a[0] for a in AUDIT]; pooled = [a[3] for a in AUDIT]
    y = list(range(len(AUDIT)))[::-1]
    PI3 = 4.93                                                                               
    ax.axvspan(DAPO_MEAN - PI3,      DAPO_MEAN + PI3,      color=BAND2, lw=0, zorder=0)
    ax.axvspan(DAPO_MEAN - DAPO_SD,  DAPO_MEAN + DAPO_SD,  color=BAND1, lw=0, zorder=0)
    ax.axvline(DAPO_MEAN, color=INK, lw=0.9, zorder=1)
    for xx in (DAPO_MEAN - PI_HALF, DAPO_MEAN + PI_HALF):
        ax.axvline(xx, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    for name, yi, p in zip(names, y, pooled):
        if name == "FIPO-K":                                                                                  
            ax.plot(p, yi, "D", ms=5.5, mfc=MUTED, mec="white", mew=0.8, zorder=3); continue
        out = abs(p - DAPO_MEAN) > PI_HALF
        ax.plot(p, yi, "o", ms=5.5, mfc="white" if out else BLUE, mec=VERM if out else "white", mew=1.2 if out else 0.8, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlabel("pooled AIME 2024+2025, mean@128 at steps 320/350/380 (%)")
    ax.set_xlim(20.2, 30.4); ax.grid(axis="y", visible=False)
    handles = [Patch(color=BAND1, label="DAPO mean $\\pm$ 1 seed SD"),
               Patch(color=BAND2, label="95% prediction interval, 3 seeds ($\\pm$4.9)"),
               Line2D([0],[0], color=MUTED, lw=0.8, ls=(0, (4, 3)), label="pooled-variance interval ($\\pm$2.4)"),
               Line2D([0],[0], marker="o", color="none", mfc="white", mec=VERM, mew=1.2, ms=5.5, label="outside dashed"),
               Line2D([0],[0], marker="D", color="none", mfc=MUTED, mec="white", ms=5.5, label="FIPO-K, own window (matched diff. 0.00)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=1.6, columnspacing=1.2, fontsize=7.5)
    save(fig, "fig_seed_band.pdf")

def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: -xs[i]); r = [0]*len(xs)
    for k, i in enumerate(order): r[i] = k + 1
    return r

def fig2_rank_inversion():
    runs = [a for a in AUDIT if a[0] != "FIPO-K"]
    r24, r25 = rank([a[1] for a in runs]), rank([a[2] for a in runs])
    n = len(runs)
    rho = 1 - 6*sum((a-b)**2 for a, b in zip(r24, r25)) / (n*(n*n-1))
    fig, ax = plt.subplots(figsize=(3.6, 3.5))
    ax.plot([0.5, n+0.5], [0.5, n+0.5], "--", color=MUTED, lw=0.8, zorder=1)
    for (name, *_), a, b in zip(runs, r24, r25):
        ax.plot(a, b, "o", ms=5, mfc=BLUE, mec="white", mew=0.8, zorder=3)
        dx, dy, ha = 4, 3, "left"
        if b <= 1: dy = -9
        if a <= 2: dx, ha = -4, "right"
        ax.annotate(name, (a, b), fontsize=6.5, xytext=(dx, dy), textcoords="offset points", color=INK, ha=ha)
    ax.set_xlabel("rank on AIME 2024 (1 = best)"); ax.set_ylabel("rank on AIME 2025 (1 = best)")
    ax.set_xlim(0.5, n+0.5); ax.set_ylim(0.5, n+0.5); ax.invert_xaxis(); ax.invert_yaxis()
    ax.set_xticks(range(1, n+1, 2)); ax.set_yticks(range(1, n+1, 2))
    ax.set_title(f"Spearman $\\rho$ = {rho:+.2f} over {n} single-seed runs")
    ax.set_aspect("equal")
    save(fig, "fig_rank_inversion.pdf")

def fig3_mixed_rate():
    obs = json.load(open(OBS))
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    def line(key, color, label=None, lw=1.2, alpha=1.0, z=3):
        pts = obs[key]; ax.plot([g for g, _ in pts], [r for _, r in pts], color=color, lw=lw, alpha=alpha, label=label, zorder=z)
    for i, k in enumerate(("math7b_g128_s1", "math7b_g128_s2", "math7b_g128_s3")):
        line(k, BLUE, "Qwen2.5-Math-7B, gen 128 (three seeds)" if i == 0 else None, lw=1.0, alpha=0.75)
    line("math7b_g96",   VERM,  "Qwen2.5-Math-7B, gen 96")
    line("gen7b_g256",   PURP,  "Qwen2.5-7B (general), gen 256")
    line("math15b_g256", GREEN, "Qwen2.5-Math-1.5B, gen 256")
    for yv, col, lab in ((25, INK, "64 / 256 = 25%: the discard statistic"), (50, BLUE, "64 / 128"), (100*64/96, VERM, "64 / 96")):
        ax.axhline(yv, color=col, lw=0.8, ls=(0, (3, 2)), zorder=2)
        ax.text(394, yv + 1.5, lab, fontsize=7, color=col, ha="right", va="bottom",
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"), zorder=5)
    ax.set_xlabel("training step"); ax.set_ylabel("mixed-outcome groups (% of generated)")
    ax.set_xlim(0, 400); ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=1.8, columnspacing=1.5)
    save(fig, "fig_mixed_rate.pdf")

def fig4_gen_batch():
    gb = ["96", "128", "256"]; roll = [96*1.261, 128*1.000, 256*1.000]; frac2 = [26.1, 0.0, 0.0]
    colors = [VERM, BLUE, INK]
    fig, ax = plt.subplots(figsize=(3.9, 3.0))
    bars = ax.bar(gb, roll, color=colors, width=0.58, zorder=3)
    for b, r, f in zip(bars, roll, frac2):
        ax.text(b.get_x()+b.get_width()/2, r+6, f"{r:.0f} prompts / step", ha="center", fontsize=8, color=INK)
        ax.text(b.get_x()+b.get_width()/2, -22, f"second batch at\n{f:.0f}% of steps", ha="center", va="top", fontsize=7.5, color=INK)
    pm = 64/0.70
    ax.axhline(pm, color=INK, lw=1.1, ls=(0, (4, 2)), zorder=4)
    ax.annotate("predicted minimum\n$64 / 0.70 \\approx 91$ prompts", xy=(1.5, pm), xycoords="data",
                xytext=(1.5, 205), textcoords="data", fontsize=8, color=INK, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.7, shrinkB=2),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb", lw=0.6))
    ax.set_ylim(0, 300); ax.set_ylabel("prompts generated per step")
    ax.set_xlabel("generation batch (prompts)", labelpad=30)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_gen_batch.pdf")

def fig5_dose_response():
                                                                                            
    stdw = [0.0, 0.048, 0.119, 0.239]; a25 = [17.65, 17.20, 14.77, 13.34]; a24 = [32.96, 31.39, 33.32, 32.69]
    k25 = [(0.119, 14.61), (0.239, 12.42)]; k24 = [(0.119, 30.44), (0.239, 34.94)]
    SD25, SD24 = 1.16, 1.68
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.8), sharex=True)
    for ax, y, sd, kk, t in ((axs[0], a25, SD25, k25, "AIME 2025"), (axs[1], a24, SD24, k24, "AIME 2024")):
        ax.axhspan(y[0]-sd, y[0]+sd, color=BAND1, lw=0, zorder=0)
        ax.plot(stdw, y, "-", color=INK, lw=1.1, zorder=2)
        ax.plot(stdw, y, "o", ms=5.5, mfc=BLUE, mec="white", mew=0.8, zorder=3)
        ax.plot([k[0] for k in kk], [k[1] for k in kk], "s", ms=5.5, mfc=VERM, mec="white", mew=0.8, zorder=3)
        ax.set_title(t); ax.set_xlabel("spread of advantage weights, std($w$)")
        ax.set_xticks([0, 0.1, 0.2])
    axs[0].set_ylabel("mean@128 accuracy (%)")
    handles = [Patch(color=BAND1, label="DAPO 3-seed mean $\\pm$ 1 SD"),
               Line2D([0],[0], marker="o", color=INK, lw=1.1, mfc=BLUE, mec="white", ms=5.5, label="PSO v5, uncontrolled"),
               Line2D([0],[0], marker="s", color="none", mfc=VERM, mec="white", ms=5.5, label="PSO v5 + KL-targeted step")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3, handlelength=1.6, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig_dose_response.pdf")

if __name__ == "__main__":
    fig1_seed_band(); fig2_rank_inversion(); fig3_mixed_rate(); fig4_gen_batch(); fig5_dose_response()
    print("wrote:", sorted(os.listdir(OUT)))
