import numpy as np
import matplotlib.pyplot as plt

from mds_common import load_data, make_mds, metric_mds, quality, IMAGES_DIR

OUT_FILE = IMAGES_DIR / "5_mds_variantai_stulpeline.png"

K = 15
METRICS = {                      # stulpelis quality() rezultate -> užrašas ašyje
    f"T (k={K})":     "T",
    f"C (k={K})":     "C",
    f"|T-C| (k={K})": "|T − C|",
    "rho":            "rho",
    "Stress-1":       "Stress-1",
}

COLORS = {"Metrinis": "#ffd166", "Nemetrinis": "#8ecae6"}

TEXT_COLOR = "white"      # tamsiai skaidrei; šviesiai – "black"
TRANSPARENT = True        # fonas permatomas, kad matytųsi skaidrės spalva

STYLE = {
    "font.size": 16,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 14,
    "legend.fontsize": 16,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.edgecolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "font.weight": "bold",
    "axes.labelweight": "bold",
}

# ---- Projekcijos: tie patys nustatymai kaip ataskaitos skripte ----
X, y, dX = load_data()

Y_m, _ = metric_mds(X, 2)

nm = make_mds(2, metric=False, n_init=1)
Y_nm = nm.fit_transform(X, init=Y_m)

q = {
    "Metrinis":   quality(X, Y_m, dX, scale=True),
    "Nemetrinis": quality(X, Y_nm, dX, scale=True),
}
VALUES = {name: [round(res[c], 3) for c in METRICS] for name, res in q.items()}

# ---- Diagrama ----
x = np.arange(len(METRICS))
width = 0.38
y_max = max(1.1, 1.1 * max(max(v) for v in VALUES.values()))

with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(13.33, 6))

    for i, (name, vals) in enumerate(VALUES.items()):
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, color=COLORS[name], label=name, zorder=2)
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in vals],
                     padding=4, fontsize=14, color=TEXT_COLOR)

    ax.set_xticks(x, list(METRICS.values()))
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, y_max)
    ax.set_ylabel("Reikšmė")
    ax.yaxis.grid(True, alpha=0.25, color=TEXT_COLOR)
    ax.set_axisbelow(True)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False)
    ax.text(1.0, -0.12, f"T, C, |T − C|: k = {K}",
            transform=ax.transAxes, ha="right", va="top", fontsize=13)

    if TRANSPARENT:
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight", transparent=TRANSPARENT)
    plt.show()