import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

CSV = ["A24_cleaned.csv"]
LABEL = "class"
IMAGES_DIR = Path(__file__).resolve().parent / "images"

DROP_COLS = []

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

for file in CSV:
    df = pd.read_csv(file)
    y = df[LABEL]
    X = df.drop(columns=[LABEL] + DROP_COLS).copy()

    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA()
    Z = pca.fit_transform(X_scaled)

    weights = pd.DataFrame(pca.components_[:2].T, index=X.columns, columns=["PK1", "PK2"])
    print(weights.sort_values("PK1", key=abs, ascending=False).round(3).to_string())

    evr = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ax = axes[0]
    k = np.arange(1, len(evr) + 1)
    cum = evr.cumsum()

    ax.bar(k, evr, alpha=0.7, label="Atskira komponente")
    ax.plot(k, cum, "o-", color="black", label="Suma")

    thresholds = {0.80: "tab:green", 0.85: "tab:orange", 0.90: "tab:red"}
    for thr, color in thresholds.items():
        ax.axhline(thr, ls="--", color=color, lw=1, label=f"{int(thr*100)} % riba")
        n_comp = int(np.argmax(cum >= thr)) + 1
        ax.axvline(n_comp, ls=":", color=color, lw=0.8)
        ax.annotate(f"{n_comp}", xy=(n_comp, thr), xytext=(3, 3),
                    textcoords="offset points", color=color, fontsize=9)

    ax.set_xlabel("Komponenciu skaicius")
    ax.set_ylabel("Paaiskintas variacijos santykis")
    ax.set_title("Paaiskintos variacijos grafikas (scree plot)")
    ax.legend()

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
    XLIM = (-5, 12)

    def plot_pca_scores(ax, Z, y, evr, *, clip_outlier=True, xlim=XLIM):
        ax.set_aspect("equal", adjustable="box")

        counts = y.value_counts()
        order = [c for c in counts.index if c not in TOP_CLASSES]
        order += [c for c in TOP_CLASSES if c in counts.index]

        for cls in order:
            m = (y == cls).values
            ax.scatter(
                Z[m, 0], Z[m, 1],
                s=7, alpha=0.75, linewidths=0,
                color=CLASS_COLORS.get(cls, "gray"),
                label=cls,
            )

        if clip_outlier:
            ax.set_xlim(*xlim)
            out = (Z[:, 0] < xlim[0]) | (Z[:, 0] > xlim[1])
            if out.any():
                print(f"Uz vaizdo ribu: {out.sum()} taskas(-ai), "
                    f"PK1 = {Z[out, 0].round(1).tolist()}")

        ax.set_xlabel(f"PK1 ({evr[0]:.1%})")
        ax.set_ylabel(f"PK2 ({evr[1]:.1%})")
        ax.set_title("Imtys pagrindiniu komponenciu erdveje")

        handles, labels = ax.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles)))
        ax.legend(
            handles, labels,
            markerscale=3, fontsize=8,
            loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0,
        )

    plot_pca_scores(axes[1], Z, y, evr, clip_outlier=True)
    fig.savefig(IMAGES_DIR / "pca_scores.png", dpi=200, bbox_inches="tight")

    ax = axes[2]
    load = pca.components_[:2].T
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    for name, (a, b) in zip(X.columns, load):
        ax.arrow(0, 0, a, b, head_width=0.01, color="tab:blue", alpha=0.7)
        ax.text(a * 1.08, b * 1.08, name, fontsize=7)
    ax.set_xlabel("PK1 apkrova")
    ax.set_ylabel("PK2 apkrova")
    ax.set_title("Kurie kintamieji lemia PK1 / PK2")

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / f"{Path(file).name}.png", dpi=150)
    plt.show()

    print("Kaupiamoji paaiskintoji dispersija:")
    print(pd.Series(evr.cumsum(), index=[f"PK{i}" for i in k]).round(3).to_string())
