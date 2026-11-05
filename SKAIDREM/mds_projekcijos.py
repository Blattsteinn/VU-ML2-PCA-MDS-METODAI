"""

Metrinio ir nemetrinio MDS palyginimas (d = 2)

"""

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from mds_common import (load_data, make_mds, metric_mds, quality, K_LIST,
                        IMAGES_DIR, CSV_DIR)

# ---------------- Config ----------------
D          = 2
SEEDS      = [0, 1, 2, 3, 4]        # stabilumo tyrimui
N_JOBS     = -1
XLIM       = (-9, 9)                # tik grafikams (kaip 5 skyriuje)
YLIM       = (-7, 11)
OUT_FIG    = IMAGES_DIR / "mds_8_metric_vs_nonmetric.png"
OUT_FIG_AL = IMAGES_DIR / "mds_8_aligned_overlay.png"
OUT_CSV    = CSV_DIR / "mds_8_metric_vs_nonmetric.csv"
OUT_STAB   = CSV_DIR / "mds_8_stabilumas.csv"

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


def fit(X, metric, init=None, seed=42, n_init=None):
    """init=None -> atsitiktine inicializacija; init=masyvas -> startas nuo jo (n_init = 1)."""
    if init is None:
        kw = {} if n_init is None else {"n_init": n_init}
        m = make_mds(D, metric=metric, seed=seed, **kw)
        Y = m.fit_transform(X)
    else:
        m = make_mds(D, metric=metric, n_init=1, seed=seed)
        Y = m.fit_transform(X, init=init)
    return Y, int(m.n_iter_)


def run_seed_any(X, metric, seed):
    """Vienas paleidimas su atsitiktine inicializacija (metrinis arba nemetrinis)."""
    return fit(X, metric, init=None, seed=seed, n_init=1)


def plot_scores(ax, Y, y, title):
    ax.set_aspect("equal", adjustable="box")
    counts = y.value_counts()
    order = [c for c in counts.index if c not in TOP_CLASSES]
    order += [c for c in TOP_CLASSES if c in counts.index]
    for cls in order:
        m = (y == cls).values
        ax.scatter(Y[m, 0], Y[m, 1], s=7, alpha=0.75, linewidths=0,
                   color=CLASS_COLORS.get(cls, "gray"), label=cls)
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title(title)


def normalize(Y):
    """Centruoja ir sunormuoja mastelį (Frobenijaus norma = 1) -> grafikai palyginami."""
    Y = Y - Y.mean(axis=0)
    return Y / np.linalg.norm(Y)


def main():
    X, y, dX = load_data()
    y = pd.Series(y).reset_index(drop=True)
    print(f"Objektu: {X.shape[0]}, pozymiu: {X.shape[1]}\n")

    # ------------------------------------------------------------------
    # 1. Projekcijos
    # ------------------------------------------------------------------
    t = time.time()
    Y_m, it_m = metric_mds(X, D)                                  # metrinis (cache)
    Y_nm_rand, it_nr = fit(X, metric=False)                       # nemetrinis, atsitiktine inic.
    Y_nm_init, it_ni = fit(X, metric=False, init=Y_m)             # nemetrinis, metrinio inic.
    print(f"MDS paleidimai: {time.time() - t:.1f} s\n")

    variants = {
        "Metrinis":                          (Y_m,       it_m),
        "Nemetrinis (atsitiktine inic.)":    (Y_nm_rand, it_nr),
        "Nemetrinis (metrinio MDS inic.)":   (Y_nm_init, it_ni),
    }

    # ------------------------------------------------------------------
    # 2. Kokybes matai (scale=True visiems -> Stress-1 su optimaliu alpha,
    #    nes nemetrinio MDS mastelis laisvas)
    # ------------------------------------------------------------------
    rows = [{"Variantas": name, **quality(X, Y, dX, scale=True), "iteracijos": n}
            for name, (Y, n) in variants.items()]
    tab = pd.DataFrame(rows).set_index("Variantas")
    tab.round(4).to_csv(OUT_CSV)

    pd.set_option("display.width", 220)
    tc = [f"{m} (k={k})" for k in K_LIST for m in ("T", "C")]
    print("Lokali struktura (d = 2):")
    print(tab[tc].round(3).to_string())
    print("\nGlobali struktura (d = 2):")
    glob = ["|T-C| (k=15)", "rho", "Stress-1", "alpha", "iteracijos"]
    print(tab[glob].round(3).to_string())

    # Skirtumai nuo metrinio
    base = tab.loc["Metrinis"]
    diff = (tab.drop(index="Metrinis")[tc + ["rho", "Stress-1"]] - base[tc + ["rho", "Stress-1"]])
    print("\nSkirtumas (nemetrinis - metrinis):")
    print(diff.round(3).to_string())

    # ------------------------------------------------------------------
    # 3. Konfiguraciju panasumas
    # ------------------------------------------------------------------
    print("\nKonfiguraciju panasumas (metrinis vs nemetrinis):")
    for name in list(variants)[1:]:
        Y = variants[name][0]
        _, _, disparity = procrustes(Y_m, Y)          # 0 = identiski po posukio/mastelio
        r = spearmanr(pdist(Y_m), pdist(Y))[0]        # ar sutampa atstumu tvarka
        print(f"  {name}: Procrustes disparity = {disparity:.3f}, "
              f"Spearman(d_metr, d_nemetr) = {r:.3f}")

    # ------------------------------------------------------------------
    # 4. Grafikai
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    for ax, (name, (Y, _)) in zip(axes, variants.items()):
        plot_scores(ax, Y, y, name)
    handles, labels = axes[0].get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles)))
    axes[-1].legend(handles, labels, markerscale=3, fontsize=8, loc="upper left",
                    bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.tight_layout()
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    plt.show()

    # Sulygintos (Procrustes) projekcijos ant vieno grafiko: rodo, kur taskai pasislenka
    A, B, _ = procrustes(Y_m, Y_nm_rand)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_aspect("equal", adjustable="box")
    ax.scatter(A[:, 0], A[:, 1], s=6, c="tab:blue", alpha=0.5, label="Metrinis")
    ax.scatter(B[:, 0], B[:, 1], s=6, c="tab:red", alpha=0.5, label="Nemetrinis")
    ax.set_title("Sulygintos projekcijos (Procrustes)")
    ax.legend(markerscale=3)
    fig.savefig(OUT_FIG_AL, dpi=150, bbox_inches="tight")
    plt.show()

    # ------------------------------------------------------------------
    # 5. Stabilumas: abu metodai, tos pačios sėklos, atsitiktine inicializacija
    # ------------------------------------------------------------------
    print(f"\nStabilumas (seklos {SEEDS}, n_init = 1, atsitiktine inic.)")
    jobs = [(metric, s) for metric in (True, False) for s in SEEDS]
    t = time.time()
    res = Parallel(n_jobs=N_JOBS)(delayed(run_seed_any)(X, m, s) for m, s in jobs)
    print(f"Paleidimai: {time.time() - t:.1f} s")

    stab = pd.DataFrame([
        {"metodas": "Metrinis" if m else "Nemetrinis", "sekla": s,
         **quality(X, Y, dX, scale=True), "iteracijos": n}
        for (m, s), (Y, n) in zip(jobs, res)
    ])
    stab.round(4).to_csv(OUT_STAB, index=False)

    show = ["T (k=15)", "C (k=15)", "rho", "Stress-1"]
    agg = stab.groupby("metodas")[show].agg(["mean", "std"])
    summary = pd.DataFrame({
        m: [f"{agg.loc[g, (m, 'mean')]:.3f} +- {agg.loc[g, (m, 'std')]:.3f}" for g in agg.index]
        for m in show}, index=agg.index)
    print("\nVidurkis +- std:")
    print(summary.to_string())


if __name__ == "__main__":
    main()