"""

Rezultatai ir parametru analize (MDS)

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from mds_common import load_data, make_mds, metric_mds, quality, K_LIST, IMAGES_DIR

# ---------------- Config ----------------
# Del dideliu isskirciu sumazinam siek tiek masteli
XLIM = (-9, 9)
YLIM = (-7, 11)

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
# -----------------------------------------------


def plot_mds_scores(ax, Y, y, title, xlim=XLIM, ylim=YLIM):
    ax.set_aspect("equal", adjustable="box")
    counts = y.value_counts()
    order = [c for c in counts.index if c not in TOP_CLASSES]
    order += [c for c in TOP_CLASSES if c in counts.index]

    for cls in order:
        m = (y == cls).values
        ax.scatter(Y[m, 0], Y[m, 1], s=7, alpha=0.75, linewidths=0,
                   color=CLASS_COLORS.get(cls, "gray"), label=cls)

    out = np.zeros(len(Y), dtype=bool)
    if xlim is not None:
        ax.set_xlim(*xlim)
        out |= (Y[:, 0] < xlim[0]) | (Y[:, 0] > xlim[1])
    if ylim is not None:
        ax.set_ylim(*ylim)
        out |= (Y[:, 1] < ylim[0]) | (Y[:, 1] > ylim[1])

    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles)))
    ax.legend(handles, labels, markerscale=3, fontsize=8,
              loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)


X, y, dX = load_data()
rows, plots = [], []

# ---- Bazine metrine projekcija ----
Y_m, it = metric_mds(X, 2)
rows.append({"Variantas": "Metrinis", **quality(X, Y_m, dX, scale=True), "iteracijos": it})
plots.append((Y_m, "Metrinis MDS"))

fig, ax = plt.subplots(figsize=(8, 6))
plot_mds_scores(ax, Y_m, y, "Metrinis MDS (SMACOF), d = 2")
fig.savefig(IMAGES_DIR / "mds_5_scores.png", dpi=200, bbox_inches="tight")
plt.show()

# ---- Nemetrinis, inicializuotas metriniu sprendiniu ----
nm = make_mds(2, metric=False, n_init=1)
Y_nm = nm.fit_transform(X, init=Y_m)
rows.append({"Variantas": "Nemetrinis, metrinio MDS inic.",
             **quality(X, Y_nm, dX, scale=True), "iteracijos": nm.n_iter_})
plots.append((Y_nm, "Nemetrinis (metrinio MDS inic.)"))


# ---- Lenteles ----
tab = pd.DataFrame(rows).set_index("Variantas")
pd.set_option("display.width", 200)

tc = [f"{m} (k={k})" for k in K_LIST for m in ("T", "C")]
print("\nLokali struktura (d = 2):")
print(tab[tc].round(3).to_string())
print("\nGlobali struktura (d = 2):")
print(tab[["|T-C| (k=15)", "rho", "Stress-1", "alpha", "iteracijos"]].round(3).to_string())

# ---- Variantai salia vienas kito ----
fig, axes = plt.subplots(1, len(plots), figsize=(7.5 * len(plots), 6))
for ax, (Y_, t) in zip(np.atleast_1d(axes), plots):
    plot_mds_scores(ax, Y_, y, t)
plt.tight_layout()
fig.savefig(IMAGES_DIR / "mds_5_variants.png", dpi=150, bbox_inches="tight")
plt.show()