from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.manifold import TSNE, trustworthiness

PERPLEXITY = 50
K = 15
SEED = 0

IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(exist_ok=True)

AXIS_PADDING = 0.05

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


def global_metrics(X_original, X_reduced):
    d_original = pdist(X_original, metric="euclidean")
    d_reduced = pdist(X_reduced, metric="euclidean")

    rho = spearmanr(d_original, d_reduced).statistic
    scale = np.dot(d_original, d_reduced) / np.dot(d_reduced, d_reduced)
    stress = np.sqrt(
        np.sum((d_original - scale * d_reduced) ** 2) / np.sum(d_original ** 2)
    )
    return rho, stress


def quality(X_original, X_proj, k=K):
    rho, stress = global_metrics(X_original, X_proj)
    return {
        "T": trustworthiness(X_original, X_proj, n_neighbors=k),
        "C": trustworthiness(X_proj, X_original, n_neighbors=k),
        "rho": rho,
        "Stress-1": stress,
    }


def run_tsne(X):
    return TSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        init="pca",
        random_state=SEED,
    ).fit_transform(X)


def common_limits(projections, padding=AXIS_PADDING):
    stacked = np.vstack(projections)
    lo, hi = stacked.min(axis=0), stacked.max(axis=0)
    pad = (hi - lo) * padding
    return (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])


def plot_tsne_scores(ax, Y, y, title, xlim, ylim):
    counts = y.value_counts()
    order = [c for c in counts.index if c not in TOP_CLASSES]
    order += [c for c in TOP_CLASSES if c in counts.index]

    for cls in order:
        m = (y == cls).values
        ax.scatter(Y[m, 0], Y[m, 1], s=7, alpha=0.75, linewidths=0,
                   color=CLASS_COLORS.get(cls, "gray"))

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)


df_A = pd.read_csv("A24_cleaned.csv")
df_B = pd.read_csv("A24_no_redundant.csv")
df_C = pd.read_csv("A24_no_outliers.csv")

rows_C = df_C["index_new"].to_numpy()
df_C = df_C.drop(columns="index_new").reset_index(drop=True)

X_A = df_A.drop(columns="class")
X_B = df_B.drop(columns="class")
X_C = df_C.drop(columns="class")

P_A = run_tsne(X_A)
P_B = run_tsne(X_B)
P_C = run_tsne(X_C)

q_A = pd.Series(quality(X_A, P_A))
q_B = pd.Series(quality(X_B, P_B))
q_C = pd.Series(quality(X_C, P_C))
q_A_on_C = pd.Series(quality(X_C, P_A[rows_C]))

absolute = pd.DataFrame({"A": q_A, "B": q_B, "C": q_C, "A (C eilutės)": q_A_on_C}).T
print(f"Kokybės matai (perplexity = {PERPLEXITY}, k = {K}):")
print(absolute.to_string(float_format="{:.4f}".format))

delta = pd.DataFrame({"A → B": q_B - q_A, "A → C": q_C - q_A_on_C}).T
delta.columns = ["ΔT", "ΔC", "Δrho", "ΔStress-1"]
delta.index.name = "Pokytis"

print(f"\nKokybinių matų pokytis tarp skirtingų aibių (k = {K}):")
print(delta.to_string(float_format="{:.3f}".format))

plots = [
    (P_A, df_A["class"], "A"),
    (P_B, df_B["class"], "B (be koreliuotų požymių)"),
    (P_C, df_C["class"], "C (be išskirčių)"),
]
xlim, ylim = common_limits([P for P, _, _ in plots])

aspect = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
fig, axes = plt.subplots(1, len(plots), figsize=(6 * len(plots), 6 * aspect + 1.6))
for ax, (P, labels, t) in zip(axes, plots):
    plot_tsne_scores(ax, P, labels, t, xlim, ylim)

present = sorted(set().union(*(set(labels) for _, labels, _ in plots)))
handles = [
    Line2D([0], [0], marker="o", linestyle="", markersize=7,
           markerfacecolor=CLASS_COLORS.get(cls, "gray"), markeredgewidth=0, label=cls)
    for cls in present
]
fig.suptitle(f"t-SNE projekcijos, perplexity = {PERPLEXITY}")
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0),
           ncol=len(handles), fontsize=9, frameon=True)

fig.savefig(IMAGES_DIR / "tsne_variants.png", dpi=150, bbox_inches="tight")
plt.show()

for (P, labels, t), name in zip(plots, ["A", "B", "C"]):
    fig_single, ax = plt.subplots(figsize=(7, 7 * aspect + 1.4))
    plot_tsne_scores(ax, P, labels, f"{t}, perplexity = {PERPLEXITY}", xlim, ylim)
    fig_single.tight_layout(rect=(0, 0.12, 1, 1))
    fig_single.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0),
                      ncol=4, fontsize=9, frameon=True)
    fig_single.savefig(IMAGES_DIR / f"tsne_{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig_single)
