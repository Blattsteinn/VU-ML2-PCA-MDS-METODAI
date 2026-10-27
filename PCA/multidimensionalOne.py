import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

FILE, LABEL = "A24_cleaned.csv", "class"
OUTPUT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = OUTPUT_DIR / "images"
DIMS = [2, 3, 5, 10, 15]
KS = [5, 15, 50]

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(FILE)
y = df[LABEL]
X = StandardScaler().fit_transform(df.drop(columns=[LABEL]))
DIMS.append(X.shape[1])
DX = pdist(X)

rows = []
for d in DIMS:
    Y = PCA(n_components=d).fit_transform(X)
    DY = pdist(Y)
    rho = spearmanr(DX, DY).correlation
    stress = np.sqrt(((DX - DY) ** 2).sum() / (DX ** 2).sum())
    for k in KS:
        rows.append(dict(d=d, k=k,
                         T=trustworthiness(X, Y, n_neighbors=k),
                         C=trustworthiness(Y, X, n_neighbors=k),
                         rho=rho, stress=stress))
res = pd.DataFrame(rows)
print(res.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for k in KS:
    s = res[res.k == k]
    axes[0].plot(s.d, s["T"], "o-", label=f"k={k}")
    axes[1].plot(s.d, s["C"], "o-", label=f"k={k}")
axes[0].set_title("Trustworthiness vs d"); axes[1].set_title("Continuity vs d")
for ax in axes[:2]:
    ax.set_xlabel("d"); ax.legend(); ax.grid(alpha=0.3)

s = res[res.k == KS[0]]
axes[2].plot(s.d, s.rho, "o-", label="Spearman rho")
axes[2].plot(s.d, s.stress, "s-", label="Stress-1")
axes[2].set_title("Globali struktura vs d"); axes[2].set_xlabel("d")
axes[2].legend(); axes[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(IMAGES_DIR / "pca_metrics_vs_d.png", dpi=150); plt.show()

pca = PCA().fit(X)
cum = pca.explained_variance_ratio_.cumsum()
print("Kaiser (eigenvalue > 1):", (pca.explained_variance_ > 1).sum())
for t in [0.80, 0.90, 0.95]:
    print(f"{t:.0%} reached at PC{np.argmax(cum >= t) + 1}")

Z = pca.transform(X)
evr = pca.explained_variance_ratio_
pairs = [(0, 1), (0, 2), (1, 2)]
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, (i, j) in zip(axes, pairs):
    for cls in sorted(y.unique()):
        m = (y == cls).values
        ax.scatter(Z[m, i], Z[m, j], s=10, alpha=0.5, label=cls)
    ax.set_xlabel(f"PC{i+1} ({evr[i]:.1%})")
    ax.set_ylabel(f"PC{j+1} ({evr[j]:.1%})")
axes[0].legend(markerscale=2, fontsize=8)
plt.tight_layout(); plt.savefig(IMAGES_DIR / "pca_pc1_pc3.png", dpi=150); plt.show()
