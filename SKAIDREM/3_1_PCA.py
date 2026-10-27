import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

CSV = "A24_cleaned.csv"
LABEL = "class"
DROP_COLS = []
IMAGES_DIR = Path(__file__).resolve().parent / "images"
OUT_FILE = "4_scree_plot.png"

THRESHOLDS = {0.80: "#2e7d32", 0.85: "#ef6c00", 0.90: "#c62828"}

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 24,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)
X = df.drop(columns=[LABEL] + DROP_COLS)
X_scaled = StandardScaler().fit_transform(X)

evr = PCA().fit(X_scaled).explained_variance_ratio_
cum = evr.cumsum()
k = np.arange(1, len(evr) + 1)

fig, ax = plt.subplots(figsize=(13.33, 7.5))

ax.bar(k, evr, color="#90a4ae", width=0.75, label="Atskira komponentė")
ax.plot(k, cum, "o-", color="#263238", lw=2.5, ms=7, label="Kaupiamoji dalis")

for thr, color in THRESHOLDS.items():
    d = int(np.argmax(cum >= thr)) + 1
    ax.axhline(thr, ls="--", color=color, lw=2)
    ax.axvline(d, ls=":", color=color, lw=2,
               label=f"{int(thr * 100)} % riba → d = {d}")

ax.set_xlim(0.3, len(evr) + 0.7)
ax.set_ylim(0, 1.03)
ax.set_xticks(k)
ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax.grid(axis="y", alpha=0.3)

ax.set_xlabel("Komponenčių skaičius d")
ax.set_ylabel("Paaiškintos dispersijos dalis")
ax.legend(loc="center right", frameon=False)

fig.tight_layout()
fig.savefig(IMAGES_DIR / OUT_FILE, dpi=300, bbox_inches="tight")
plt.show()
