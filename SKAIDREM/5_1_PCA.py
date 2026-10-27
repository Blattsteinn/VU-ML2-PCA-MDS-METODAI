import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

CSV = "A24_cleaned.csv"
LABEL = "class"
DROP_COLS = []
IMAGES_DIR = Path(__file__).resolve().parent / "images"
OUT_FILE = IMAGES_DIR / "5_pca_scatter_prezentacijai.png"

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
XLIM = (-5, 12)                          # None – rodyti visus taškus
YLIM = None                              # None – pagal taškus, patekusius į XLIM

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

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)
y = df[LABEL]
X = df.drop(columns=[LABEL] + DROP_COLS)
X_scaled = StandardScaler().fit_transform(X)

pca = PCA()
Z = pca.fit_transform(X_scaled)
evr = pca.explained_variance_ratio_

counts = y.value_counts()
order = [c for c in counts.index if c not in TOP_CLASSES]
order += [c for c in TOP_CLASSES if c in counts.index]

with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(13.33, 7.5))  # 16:9 skaidrė
    ax.set_aspect("equal", adjustable="box")

    for cls in order:
        m = (y == cls).values
        ax.scatter(Z[m, 0], Z[m, 1], s=20, alpha=0.75, linewidths=0,
                   color=CLASS_COLORS.get(cls, "gray"), label=cls, zorder=2)

    ax.axhline(0, color="grey", lw=0.8, alpha=0.5, zorder=1)
    ax.axvline(0, color="grey", lw=0.8, alpha=0.5, zorder=1)

    if XLIM is not None:
        inside = (Z[:, 0] >= XLIM[0]) & (Z[:, 0] <= XLIM[1])
        if YLIM is not None:
            ylim = YLIM
        else:
            lo, hi = Z[inside, 1].min(), Z[inside, 1].max()
            pad = 0.05 * (hi - lo)
            ylim = (lo - pad, hi + pad)
        ax.set_xlim(*XLIM)
        ax.set_ylim(*ylim)
        inside &= (Z[:, 1] >= ylim[0]) & (Z[:, 1] <= ylim[1])
        n_out = int((~inside).sum())
        if n_out:
            ax.text(0.99, 0.01, f"Už vaizdo ribų: {n_out} tašk.",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=13, color="dimgray")

    ax.set_xlabel(f"PK1 ({evr[0]:.1%})")
    ax.set_ylabel(f"PK2 ({evr[1]:.1%})")

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles)))
    ax.legend(handles, labels, markerscale=2.5, frameon=False,
              loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)

    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    plt.show()