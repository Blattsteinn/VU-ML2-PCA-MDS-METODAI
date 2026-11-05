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
    X = df.drop(columns=[LABEL] + DROP_COLS).copy()

    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA()
    pca.fit(X_scaled)

    load = pca.components_[:2].T

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)

    for name, (a, b) in zip(X.columns, load):
        ax.arrow(0, 0, a, b, head_width=0.01, color="tab:blue", alpha=0.7)
        ax.text(a * 1.08, b * 1.08, name, fontsize=7)

    ax.set_xlabel("PK1 apkrova")
    ax.set_ylabel("PK2 apkrova")

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / f"{Path(file).stem}_loadings.png", dpi=150)
    plt.show()
