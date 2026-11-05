import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

IMAGES_DIR = Path(__file__).resolve().parent / "images"
OUT_FILE = "6_pca_mds_kokybe.png"

# Reikšmės iš ataskaitos 13–18 lentelių
K = [5, 15, 50]
LOCAL = {
    "PCA": {"T": [0.859, 0.860, 0.865], "C": [0.965, 0.953, 0.937]},
    "MDS": {"T": [0.887, 0.887, 0.889], "C": [0.956, 0.949, 0.939]},
}
GLOBAL = {"PCA": [0.804, 0.417], "MDS": [0.881, 0.233]}   # rho, Stress-1
GLOBAL_LABELS = ["rho", "Stress-1"]

COLORS = {"PCA": "#90caf9", "MDS": "#ffcc80"}
STYLES = {"T": ("o", "-"), "C": ("s", "--")}
FG = "black"                # tekstas ir ašys tamsiam skaidrės fonui
TRANSPARENT = True          # be fono, kad tiktų ant skaidrės

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": FG,
    "axes.labelcolor": FG,
    "axes.edgecolor": FG,
    "xtick.color": FG,
    "ytick.color": FG,
})

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.33, 6.5),
                               gridspec_kw={"width_ratios": [1.4, 1]})

# Kairė: T ir C pagal k
for method, metrics in LOCAL.items():
    for metric, values in metrics.items():
        marker, ls = STYLES[metric]
        ax1.plot(K, values, marker=marker, ls=ls, color=COLORS[method],
                 lw=2.5, ms=9, label=f"{method} {metric}")
ax1.set_xscale("log")
ax1.set_xticks(K)
ax1.set_xticklabels(K)
ax1.minorticks_off()
ax1.set_ylim(0.84, 0.98)
ax1.set_xlabel("Kaimynų skaičius k")
ax1.set_ylabel("Reikšmė")
ax1.set_title("Lokali struktūra")
ax1.grid(axis="y", alpha=0.3)
ax1.legend(loc="center right", frameon=False, ncol=2)

# Dešinė: rho ir Stress-1
x = np.arange(len(GLOBAL_LABELS))
w = 0.36
for i, (method, values) in enumerate(GLOBAL.items()):
    bars = ax2.bar(x + (i - 0.5) * w, values, w, color=COLORS[method], label=method)
    ax2.bar_label(bars, fmt="%.3f", padding=4, fontsize=15, color=FG)
ax2.set_xticks(x)
ax2.set_xticklabels(GLOBAL_LABELS)
ax2.set_ylim(0, 1.0)
ax2.set_title("Globali struktūra")
ax2.grid(axis="y", alpha=0.3)
ax2.legend(loc="upper right", frameon=False)

fig.tight_layout()
fig.savefig(IMAGES_DIR / OUT_FILE, dpi=300, bbox_inches="tight", transparent=TRANSPARENT)
plt.show()