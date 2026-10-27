"""

Projekciju jautrumo duomenu paruosimui tyrimas (PCA) A su B

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
CSV_B = BASE_DIR / "A24_no_redundant.csv"
LABEL = "class"
IMAGES_DIR = BASE_DIR / "images"

D_LOCAL = [2]
D_GLOBAL = [2]
K_LIST = [5, 15, 50]
MAKE_FIGURES = False

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


def run_pca(csv_path):
    df = pd.read_csv(csv_path)
    X = StandardScaler().fit_transform(df.drop(columns=[LABEL]).to_numpy(dtype=float))
    pca = PCA(svd_solver="full")
    Z = pca.fit_transform(X)
    return X, Z, pca.explained_variance_ratio_, df[LABEL]


def local_metrics(X_ref, Z):
    return {(d, k): (trustworthiness(X_ref, Z[:, :d], n_neighbors=k),
                     trustworthiness(Z[:, :d], X_ref, n_neighbors=k))
            for d in D_LOCAL for k in K_LIST}


def global_metrics(X_ref, Z):
    d_ref = pdist(X_ref)
    out = {}
    for d in D_GLOBAL:
        d_Y = pdist(Z[:, :d])
        rho = spearmanr(d_ref, d_Y)[0]
        stress = np.sqrt(np.sum((d_ref - d_Y) ** 2) / np.sum(d_ref ** 2))
        out[d] = (rho, stress)
    return out


def table_local(mA, mO, other):
    rows = []
    for (d, k), (tA, cA) in mA.items():
        tO, cO = mO[(d, k)]
        rows.append({"d": d, "k": k,
                     "T (A)": tA, f"T ({other})": tO, "ΔT": tO - tA,
                     "C (A)": cA, f"C ({other})": cO, "ΔC": cO - cA})
    return pd.DataFrame(rows)


def table_global(gA, gO, other):
    rows = []
    for d, (rA, sA) in gA.items():
        rO, sO = gO[d]
        rows.append({"d": d,
                     "rho (A)": rA, f"rho ({other})": rO, "Δrho": rO - rA,
                     "Stress-1 (A)": sA, f"Stress-1 ({other})": sO, "ΔStress-1": sO - sA})
    return pd.DataFrame(rows)


def print_table(df, title):
    print(f"\n=== {title} ===")
    print(df.round(3).to_string(index=False))


def plot_pca_scores(ax, Z, y, evr, title):
    ax.set_aspect("equal", adjustable="box")
    counts = y.value_counts()
    order = [c for c in counts.index if c not in TOP_CLASSES]
    order += [c for c in TOP_CLASSES if c in counts.index]
    for cls in order:
        mask = (y == cls).to_numpy()
        ax.scatter(Z[mask, 0], Z[mask, 1], s=7, alpha=0.75, linewidths=0,
                   color=CLASS_COLORS.get(cls, "gray"), label=cls)
    ax.set_xlim(*XLIM)
    ax.set_xlabel(f"PK1 ({evr[0]:.1%})")
    ax.set_ylabel(f"PK2 ({evr[1]:.1%})")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles)))
    ax.legend(handles, labels, markerscale=3, fontsize=8, loc="upper left",
              bbox_to_anchor=(1.02, 1), borderaxespad=0)


def plot_scree(ax, evrs):
    for (name, evr), color in zip(evrs.items(), ("tab:blue", "tab:orange")):
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


def make_figures(res):
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    plot_scree(ax, {name: r[2] for name, r in res.items()})
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "pca_7th_section_scree_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(res), figsize=(16, 6))
    for ax, (name, (_, Z, evr, y)) in zip(axes, res.items()):
        plot_pca_scores(ax, Z, y, evr, f"{name} — PK1, PK2")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "pca_7th_section_scatter_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    res = {"A aibe": run_pca(CSV_A), "B aibe": run_pca(CSV_B)}
    (X_A, Z_A, _, y_A), (X_B, Z_B, _, y_B) = res["A aibe"], res["B aibe"]
    if len(y_A) != len(y_B) or not (y_A.to_numpy() == y_B.to_numpy()).all():
        raise ValueError("B aibes objektai ar ju tvarka nesutampa su A.")

    print_table(table_local(local_metrics(X_A, Z_A), local_metrics(X_B, Z_B), "B"),
                "7.1 A -> B: T ir C")
    print_table(table_global(global_metrics(X_A, Z_A), global_metrics(X_B, Z_B), "B"),
                "7.1 A -> B: Spirmeno rho ir Stress-1")

    if MAKE_FIGURES:
        make_figures(res)


if __name__ == "__main__":
    main()
