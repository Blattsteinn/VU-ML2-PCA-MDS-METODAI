"""

Projekciju jautrumo duomenu paruosimui tyrimas (PCA) A su C

"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
CSV_A = BASE_DIR / "A24_cleaned.csv"
CSV_C = BASE_DIR / "A24_no_outliers.csv"
LABEL = "class"
IMAGES_DIR = BASE_DIR / "images"

K_LIST = [5, 15, 50]
D_LIST = [2]

CLASS_COLORS = {
    "Bumps": "#ff9100",
    "Dirtiness": "#00d5ff",
    "K_Scratch": "#00c853",
    "Other_Faults": "#ff1744",
    "Pastry": "#aa00ff",
    "Stains": "#1f77ff",
    "Z_Scratch": "#ff00c8",
}
TOP_CLASSES = ["Stains", "K_Scratch"]
XLIM = (-5, 12)

def match_rows(X_A, y_A, X_C, y_C):
    def key(x, lab):
        return tuple(np.round(x, 9)) + (str(lab),)

    pool = {}
    for i, (x, lab) in enumerate(zip(X_A, y_A)):
        pool.setdefault(key(x, lab), []).append(i)

    idx = []
    for x, lab in zip(X_C, y_C):
        positions = pool.get(key(x, lab))
        if not positions:
            raise ValueError("C aibes eilute nerasta A aibeje – patikrink failus.")
        idx.append(positions.pop(0))
    return np.array(idx)

def stress_1(X, Y):
    dX, dY = pdist(X), pdist(Y)
    return float(np.sqrt(np.sum((dX - dY) ** 2) / np.sum(dX ** 2)))

def metrics(X_ref, Z, evr):
    res = {}
    d_ref = pdist(X_ref)
    for d in D_LIST:
        Y = Z[:, :d]
        res[(d, "var")] = float(evr[:d].sum())
        res[(d, "rho")] = float(spearmanr(d_ref, pdist(Y))[0])
        res[(d, "stress")] = stress_1(X_ref, Y)
        for k in K_LIST:
            res[(d, f"T{k}")] = trustworthiness(X_ref, Y, n_neighbors=k)
            res[(d, f"C{k}")] = trustworthiness(Y, X_ref, n_neighbors=k)
    return res

def plot_scores(ax, Z, y, evr, title, Z_removed=None):
    ax.set_aspect("equal", adjustable="box")

    if Z_removed is not None:
        ax.scatter(Z_removed[:, 0], Z_removed[:, 1], s=14, marker="x",
                   color="black", linewidths=0.6, label="Pasalinti (C)")

    counts = y.value_counts()
    order = [c for c in counts.index if c not in TOP_CLASSES]
    order += [c for c in TOP_CLASSES if c in counts.index]
    for cls in order:
        mask = (y == cls).to_numpy()
        ax.scatter(Z[mask, 0], Z[mask, 1], s=7, alpha=0.75, linewidths=0,
                   color=CLASS_COLORS.get(cls, "gray"), label=cls)

    ax.set_xlim(*XLIM)
    out = (Z[:, 0] < XLIM[0]) | (Z[:, 0] > XLIM[1])
    if out.any():
        print(f"{title}: uz PK1 grafiko ribu {out.sum()} tasku")

    ax.set_xlabel(f"PK1 ({evr[0]:.1%})")
    ax.set_ylabel(f"PK2 ({evr[1]:.1%})")
    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles)))
    ax.legend(handles, labels, markerscale=3, fontsize=8, loc="upper left",
              bbox_to_anchor=(1.02, 1), borderaxespad=0)


def plot_scree(ax, evr_A, evr_C):
    for evr, name, color in [(evr_A, "A aibe", "tab:blue"), (evr_C, "C aibe", "tab:orange")]:
        k = np.arange(1, len(evr) + 1)
        ax.plot(k, evr, "o-", color=color, markersize=4, label=f"{name} – atskira")
        ax.plot(k, evr.cumsum(), "s--", color=color, markersize=3, alpha=0.7,
                label=f"{name} – kaupiamoji")
    for threshold, color in {0.80: "tab:green", 0.85: "tab:orange", 0.90: "tab:red"}.items():
        ax.axhline(threshold, linestyle=":", color=color, linewidth=1,
                   label=f"{threshold:.0%} riba")
    ax.set_xlabel("Komponenciu skaicius")
    ax.set_ylabel("Paaiskintos dispersijos dalis")
    ax.set_title("Scree grafiku palyginimas")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

def build_tables(X_ref, Z_A_common, evr_A, Z_C, evr_C):
    mA = metrics(X_ref, Z_A_common, evr_A)
    mC = metrics(X_ref, Z_C, evr_C)

    rows = []
    for d in D_LIST:
        for k in K_LIST:
            rows.append({
                "d": d, "k": k,
                "T (A)": mA[(d, f"T{k}")], "T (C)": mC[(d, f"T{k}")],
                "ΔT": mC[(d, f"T{k}")] - mA[(d, f"T{k}")],
                "C (A)": mA[(d, f"C{k}")], "C (C)": mC[(d, f"C{k}")],
                "ΔC": mC[(d, f"C{k}")] - mA[(d, f"C{k}")],
            })
    tbl_local = pd.DataFrame(rows)

    rows = []
    for d in D_LIST:
        rows.append({
            "d": d,
            "Dispersija (A)": mA[(d, "var")], "Dispersija (C)": mC[(d, "var")],
            "rho (A)": mA[(d, "rho")], "rho (C)": mC[(d, "rho")],
            "Δrho": mC[(d, "rho")] - mA[(d, "rho")],
            "Stress-1 (A)": mA[(d, "stress")], "Stress-1 (C)": mC[(d, "stress")],
            "ΔStress-1": mC[(d, "stress")] - mA[(d, "stress")],
        })
    return tbl_local, pd.DataFrame(rows)

def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    df_A, df_C = pd.read_csv(CSV_A), pd.read_csv(CSV_C)
    features = [c for c in df_A.columns if c != LABEL]
    X_A, y_A = df_A[features].to_numpy(dtype=float), df_A[LABEL]
    X_C, y_C = df_C[features].to_numpy(dtype=float), df_C[LABEL].reset_index(drop=True)

    idx_common = match_rows(X_A, y_A, X_C, y_C)
    removed_mask = np.ones(len(df_A), dtype=bool)
    removed_mask[idx_common] = False
    print(f"A: {len(df_A)}, C: {len(df_C)}, pasalinta: {removed_mask.sum()} "
          f"({removed_mask.mean():.1%})")
    print("Pasalinta pagal klase:")
    print(y_A[removed_mask].value_counts().to_string())

    pca_A = PCA(svd_solver="full")
    Z_A = pca_A.fit_transform(StandardScaler().fit_transform(X_A))
    evr_A = pca_A.explained_variance_ratio_

    scaler_A = StandardScaler().fit(X_A)
    scaler_C = StandardScaler().fit(X_C)
    pca_C = PCA(svd_solver="full")
    Z_C = pca_C.fit_transform(scaler_C.transform(X_C))
    evr_C = pca_C.explained_variance_ratio_

    Z_A_common = Z_A[idx_common]

    for i in range(2):
        r = np.corrcoef(Z_A_common[:, i], Z_C[:, i])[0, 1]
        print(f"PK{i + 1}: koreliacija tarp A ir C koordinaciu bendriems objektams = {r:.3f}")
        if r < 0:
            Z_C[:, i] = -Z_C[:, i]

    references = {
        "A": scaler_A.transform(X_C),
        "C": scaler_C.transform(X_C),
    }
    for ref_name, X_ref in references.items():
        tbl_local, tbl_global = build_tables(X_ref, Z_A_common, evr_A, Z_C, evr_C)
        print(f"\n=== T ir C, bendri objektai, atskaita {ref_name} ===")
        print(tbl_local.round(3).to_string(index=False))
        print(f"\n=== Dispersija, rho ir Stress-1, bendri objektai, atskaita {ref_name} ===")
        print(tbl_global.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(11, 6))
    plot_scree(ax, evr_A, evr_C)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "pca_7_2_scree_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    plot_scores(axes[0], Z_A_common, y_C, evr_A, "A aibe — PK1, PK2",
                Z_removed=Z_A[removed_mask])
    plot_scores(axes[1], Z_C, y_C, evr_C, "C aibe — PK1, PK2")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "pca_7_2_scatter_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()