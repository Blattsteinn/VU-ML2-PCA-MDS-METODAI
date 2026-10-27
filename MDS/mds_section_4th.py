"""

Tinkamiausios dimensijos vertinimas (MDS)

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

from mds_common import load_data, metric_mds, quality, stress1, IMAGES_DIR, CSV_DIR

# ---------------- Cnfig ----------------
D_CURVE = [1, 2, 3, 4, 5, 7, 10]   # d reiksmes kreivei
D_TABLE = [2, 3, 5, 10]            # d reiksmes lentelems
K       = 15
OUT_FIG   = IMAGES_DIR / "mds_4_stress_nuo_d.png"
# -----------------------------------------------

X, _, dX = load_data()
curve_rows, table_rows = [], []
for d in D_CURVE:
    Y, n_iter = metric_mds(X, d)
    s1, _ = stress1(dX, pdist(Y))
    curve_rows.append({"d": d, "Stress-1 (MDS)": s1, "iteracijos": n_iter})
    if d in D_TABLE:
        q = quality(X, Y, dX, k_list=[K])
        table_rows.append({"d": d, f"T (k={K})": q[f"T (k={K})"], f"C (k={K})": q[f"C (k={K})"],
                           "|T-C|": q[f"|T-C| (k={K})"], "Spearman rho": q["rho"],
                           "Stress-1": q["Stress-1"]})
    print(f"d={d:2d}  Stress-1={s1:.3f}  iter={n_iter}")

curve = pd.DataFrame(curve_rows)
table = pd.DataFrame(table_rows)
curve["ΔStress-1 / dim."] = -curve["Stress-1 (MDS)"].diff() / curve["d"].diff()
curve["Santykinis sumazejimas / dim."] = curve["ΔStress-1 / dim."] / curve["Stress-1 (MDS)"].shift()

pd.set_option("display.float_format", "{:.3f}".format)
print("\nStress-1 priklausomybe nuo d:")
print(curve.to_string(index=False))
print(f"\nLentele (k = {K}):")
print(table.to_string(index=False))

# ---- Vizualiai ----
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(curve["d"], curve["Stress-1 (MDS)"], "o-")
for d in D_TABLE:
    v = curve.loc[curve["d"] == d, "Stress-1 (MDS)"].item()
    ax.annotate(f"{v:.3f}", (d, v), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=10)
ax.set_xlabel("Tiksline dimensija d", fontsize=14, fontweight="bold", labelpad=8)
ax.set_ylabel("Stress-1", fontsize=14, fontweight="bold", labelpad=8)
ax.set_xticks(D_CURVE)
ax.tick_params(axis="both", which="major", labelsize=12, width=1.2, length=5)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_FIG, dpi=200)
plt.show()