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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.33, 7.5))

# Kairė: lokalūs ir eiliškumo matai
ax1.plot(D, T, "o-", color=COLORS["T"], lw=2.5, ms=8, label="T")
ax1.plot(D, C, "s-", color=COLORS["C"], lw=2.5, ms=8, label="C")
ax1.plot(D, RHO, "^-", color=COLORS["rho"], lw=2.5, ms=8, label="rho")
ax1.set_ylim(0.85, 1.01)
ax1.set_title("T, C ir rho")
ax1.set_ylabel("Reikšmė")
ax1.legend(loc="lower right", frameon=False)

# Dešinė: atstumų iškraipymas
ax2.plot(D, STRESS, "o-", color=COLORS["stress"], lw=2.5, ms=8)
for d, s in zip(D, STRESS):
    ax2.annotate(f"{s:.3f}", (d, s), textcoords="offset points",
                 xytext=(8, 8), fontsize=15, color=COLORS["stress"])
ax2.set_ylim(0, 0.26)
ax2.set_title("Stress-1")

for ax in (ax1, ax2):
    ax.set_xticks(D)
    ax.set_xlim(1.5, 10.5)
    ax.set_xlabel("Dimensija d")
    ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(IMAGES_DIR / OUT_FILE, dpi=300, bbox_inches="tight")
plt.show()
