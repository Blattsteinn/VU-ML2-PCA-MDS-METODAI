import numpy as np
import matplotlib.pyplot as plt

from mds_common import load_data, metric_mds, IMAGES_DIR

OUT_FILE = IMAGES_DIR / "5_mds_scatter_prezentacijai.png"

CLASS_COLORS = {
    "Bumps":        "#ff9100",
    "Dirtiness":    "#00d5ff",
    "K_Scratch":    "#00c853",
    "Other_Faults": "#ff1744",
    "Pastry":       "#aa00ff",
    "Stains":       "#1f77ff",
    "Z_Scratch":    "#ff00c8",
}
TOP_CLASSES = ["Stains", "K_Scratch"]   # piešiamos paskutinės (viršuje)
XLIM = (-9, 9)                           # None – rodyti visus taškus
YLIM = (-7, 11)                          # None – rodyti visus taškus

STYLE = {
    "font.size": 16,
    "axes.titlesize": 22,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 15,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

X, y, dX = load_data()
Y, _ = metric_mds(X, 2)   # ta pati projekcija kaip ataskaitoje

counts = y.value_counts()
order = [c for c in counts.index if c not in TOP_CLASSES]
order += [c for c in TOP_CLASSES if c in counts.index]

with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(13.33, 7.5))  # 16:9 skaidrė
    ax.set_aspect("equal", adjustable="box")

    for cls in order:
        m = (y == cls).values
        ax.scatter(Y[m, 0], Y[m, 1], s=20, alpha=0.75, linewidths=0,
                   color=CLASS_COLORS.get(cls, "gray"), label=cls, zorder=2)

    ax.axhline(0, color="grey", lw=0.8, alpha=0.5, zorder=1)
    ax.axvline(0, color="grey", lw=0.8, alpha=0.5, zorder=1)

    out = np.zeros(len(Y), dtype=bool)
    if XLIM is not None:
        ax.set_xlim(*XLIM)
        out |= (Y[:, 0] < XLIM[0]) | (Y[:, 0] > XLIM[1])
    if YLIM is not None:
        ax.set_ylim(*YLIM)
        out |= (Y[:, 1] < YLIM[0]) | (Y[:, 1] > YLIM[1])
    if out.any():
        ax.text(0.99, 0.01, f"Už vaizdo ribų: {int(out.sum())} tašk.",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=13, color="dimgray")

    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles)))
    ax.legend(handles, labels, markerscale=2.5, frameon=False,
              loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)

    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    plt.show()