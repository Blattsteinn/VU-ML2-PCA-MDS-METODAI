"""

Projekciju kokybes ir stabilumo vertinimas (MDS)

"""

import time
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

from mds_common import (load_data, metric_mds, run_seed, quality, stress1,
                        K_LIST, MAX_ITER, CSV_DIR)

# ---------------- Config ----------------
D_LIST     = [2]
STAB_SEEDS = [0, 1, 2, 3, 4]
STAB_D     = [2]
N_JOBS     = -1
OUT_QUALITY = CSV_DIR / "mds_6_1_kokybe.csv"
OUT_STAB    = CSV_DIR / "mds_6_2_stabilumas.csv"
# -----------------------------------------------


def main():
    X, _, dX = load_data()
    print(f"Objektu: {X.shape[0]}, pozymiu: {X.shape[1]}\n")

   # --- Kokybes matai 
    print(" ------------- Metrines MDS projekcijos kokybes matai ----")
    rows = []
    for d in D_LIST:
        Y, n_iter = metric_mds(X, d)
        q = quality(X, Y, dX)
        s_pca, _ = stress1(dX, pdist(PCA(n_components=d).fit_transform(X)))
        flag = "OK" if q["Stress-1"] <= s_pca else "MDS > PCA -> SMACOF nesukonvergavo?"
        rows.append({"d": d, **q, "Stress-1 (PCA)": s_pca, "iteracijos": n_iter})

    qual = pd.DataFrame(rows)
    print("\n --- T ir C:")
    print(qual[["d"] + [f"{m} (k={k})" for k in K_LIST for m in ("T", "C")]]
          .round(3).to_string(index=False))
    print("\n --- Globali struktura:")
    print(qual[["d", "rho", "Stress-1", "Stress-1 (PCA)"]].round(3).to_string(index=False))
    print("\n ---|T − C|:")
    print(qual[["d"] + [f"|T-C| (k={k})" for k in K_LIST]].round(3).to_string(index=False))

    # --- Stabilumas
    print(f"\n ------------- Stabilumas (seklos {STAB_SEEDS}, n_init=1, atsitiktine inic.) ----")
    jobs = [(d, s) for d in STAB_D for s in STAB_SEEDS]
    t = time.time()
    results = Parallel(n_jobs=N_JOBS)(delayed(run_seed)(X, d, s) for d, s in jobs)
    print(f"SMACOF paleidimai: {time.time() - t:.1f} s")

    stab = pd.DataFrame([{"d": d, "sekla": s, **quality(X, Y, dX), "iteracijos": n_iter}
                         for (d, s), (Y, n_iter) in zip(jobs, results)])
    print("\nVisi paleidimai:")
    print(stab.drop(columns=[c for c in stab if c.startswith("|T-C|")]).round(3).to_string(index=False))

    show = ["T (k=15)", "C (k=15)", "rho", "Stress-1"]
    agg = stab.groupby("d")[show].agg(["mean", "std"])
    rng = stab.groupby("d")[show].agg(lambda v: v.max() - v.min())
    summary = pd.DataFrame({m: [f"{agg.loc[d, (m, 'mean')]:.3f} +- {agg.loc[d, (m, 'std')]:.3f}"
                                for d in agg.index] for m in show}, index=agg.index)
    print("\nVidurkis +- std:")
    print(summary.to_string())


if __name__ == "__main__":
    main()