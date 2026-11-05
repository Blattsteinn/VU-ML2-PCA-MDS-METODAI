import matplotlib.pyplot as plt
from pathlib import Path

IMAGES_DIR = Path(__file__).resolve().parent / "images"
OUT_FILE = "4_mds_dimensija.png"

# Reikšmės iš ataskaitos 5 lentelės (k = 15)
D = [2, 3, 5, 10]
T = [0.887, 0.944, 0.989, 0.999]
C = [0.949, 0.976, 0.994, 1.000]
RHO = [0.881, 0.926, 0.969, 0.998]
STRESS = [0.233, 0.155, 0.082, 0.020]

COLORS = {"T": "#1565c0", "C": "#2e7d32", "rho": "#6a1b9a", "stress": "#c62828"}

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(13.33, 7.5))

# Visi matai viename grafike
ax.plot(D, T, "o-", color=COLORS["T"], lw=2.5, ms=8, label="T")
ax.plot(D, C, "s-", color=COLORS["C"], lw=2.5, ms=8, label="C")
ax.plot(D, RHO, "^-", color=COLORS["rho"], lw=2.5, ms=8, label="rho")
ax.plot(D, STRESS, "D-", color=COLORS["stress"], lw=2.5, ms=8, label="Stress-1")

# Stress-1 reikšmių užrašai
for d, s in zip(D, STRESS):
    ax.annotate(f"{s:.3f}", (d, s), textcoords="offset points",
                xytext=(8, 8), fontsize=15, color=COLORS["stress"])

ax.set_ylim(0, 1.02)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xticks(D)
ax.set_xlim(1.5, 10.5)
ax.set_xlabel("Dimensija d")
ax.set_ylabel("Reikšmė")
ax.set_title("T, C, rho ir Stress-1")
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="center right", frameon=False)

fig.tight_layout()
fig.savefig(IMAGES_DIR / OUT_FILE, dpi=300, bbox_inches="tight")
plt.show()