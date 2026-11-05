import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

from mds_common import load_data, make_mds, metric_mds, IMAGES_DIR

OUT_METRIC  = IMAGES_DIR / "5_mds_metrinis_prezentacijai.png"
OUT_NONMET  = IMAGES_DIR / "5_mds_nemetrinis_prezentacijai.png"
OUT_BOTH    = IMAGES_DIR / "5_mds_abu_prezentacijai.png"

INIT_FROM_METRIC = True
RESCALE_NONMETRIC = True

CLASS_COLORS = {
    "Bumps":        "#ff9100",
    "Dirtiness":    "#00d5ff",
    "K_Scratch":    "#00c853",
    "Other_Faults": "#ff1744",
    "Pastry":       "#aa00ff",
    "Stains":       "#1f77ff",
    "Z_Scratch":    "#ff00c8",
}
TOP_CLASSES = ["Stains", "K_Scratch"] 
XLIM = (-9, 9)
YLIM = (-7, 11)              

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

# ---- Projekcijos ----
Y_m, _ = metric_mds(X, 2)

if INIT_FROM_METRIC:
    Y_nm = make_mds(2, metric=False, n_init=1).fit_transform(X, init=Y_m)
else:
    Y_nm = make_mds(2, metric=False).fit_transform(X)

if RESCALE_NONMETRIC:
    dY = pdist(Y_nm)
    Y_nm = Y_nm * ((dX @ dY) / (dY @ dY))

counts = y.value_counts()
order = [c for c in counts.index if c not in TOP_CLASSES]
order += [c for c in TOP_CLASSES if c in counts.index]


def draw(ax, Y, title=None, s=20, legend=True):
    ax.set_aspect("equal", adjustable="box")

    for cls in order:
        m = (y == cls).values
        ax.scatter(Y[m, 0], Y[m, 1], s=s, alpha=0.75, linewidths=0,
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
    if title:
        ax.set_title(title)

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles)))
        ax.legend(handles, labels, markerscale=2.5, frameon=False,
                  loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)


with plt.rc_context(STYLE):
    # 1) Metrinis
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    draw(ax, Y_m)
    fig.tight_layout()
    fig.savefig(OUT_METRIC, dpi=300, bbox_inches="tight")
    plt.show()

    # 2) Nemetrinis
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    draw(ax, Y_nm)
    fig.tight_layout()
    fig.savefig(OUT_NONMET, dpi=300, bbox_inches="tight")
    plt.show()

    # 3) Abu salia vienas kito
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5))
    draw(axes[0], Y_m, "Metrinis MDS", s=12, legend=False)
    draw(axes[1], Y_nm, "Nemetrinis MDS", s=12, legend=True)
    fig.tight_layout()
    fig.savefig(OUT_BOTH, dpi=300, bbox_inches="tight")
    plt.show()