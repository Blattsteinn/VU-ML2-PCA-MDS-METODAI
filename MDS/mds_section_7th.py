"""

Projekciju jautrumo duomenu paruosimui tyrimas (MDS)

"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler

from mds_common import (load_data, make_mds, metric_mds, quality, K_LIST,
                        SEED, N_INIT, MAX_ITER, DATA_PATH, CLASS_COL,
                        CACHE_DIR)

# ---------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CSV_B    = BASE_DIR / "A24_no_redundant.csv"   # B
CSV_C    = BASE_DIR / "A24_no_outliers.csv"    # C
D_LIST   = [2]
K_AC     = 15                                  # k lentelei A i C
# -----------------------------------------------


# ----------------------------------------------------------------------------
# MDS projekcijos
# ----------------------------------------------------------------------------
def report_iter(name, d, n_iter):
    note = ("  ISPEJIMAS: pasiektas MAX_ITER, SMACOF galejo nesukonvergauti!"
            if n_iter >= MAX_ITER else "")
    print(f"  {name}, d = {d}: iteraciju {n_iter}{note}")


def mds_cached(X, d, stem):
    CACHE_DIR.mkdir(exist_ok=True)
    f = CACHE_DIR / f"{stem}_n{len(X)}_metric_d{d}_seed{SEED}_ninit{N_INIT}_iter{MAX_ITER}.npz"
    if f.exists():
        z = np.load(f)
        return z["Y"], int(z["n_iter"])
    t = time.time()
    m = make_mds(d)
    Y = m.fit_transform(X)
    np.savez(f, Y=Y, n_iter=m.n_iter_)
    print(f"  [apskaiciuota MDS {stem}, d = {d}: {time.time() - t:.1f} s]")
    return Y, int(m.n_iter_)


def projections_A(X_A):
    print("A aibes projekcijos:")
    proj = {}
    for d in D_LIST:
        Y, n_iter = metric_mds(X_A, d)
        report_iter("A", d, n_iter)
        proj[d] = Y
    return proj


def projections_other(X, stem, name):
    print(f"{name} aibes projekcijos:")
    proj = {}
    for d in D_LIST:
        Y, n_iter = mds_cached(X, d, stem)
        report_iter(name, d, n_iter)
        proj[d] = Y
    return proj


# ----------------------------------------------------------------------------
# Metrikos ir lenteles
# ----------------------------------------------------------------------------
def evaluate(X_ref, proj, k_list, scale):
    dX_ref = pdist(X_ref)
    return {d: quality(X_ref, Y, dX_ref, k_list=k_list, scale=scale) for d, Y in proj.items()}


def table_T_C(qA, qO, other, k_list):
    rows = []
    for d in D_LIST:
        for k in k_list:
            row = {"d": d}
            if len(k_list) > 1:
                row["k"] = k
            for m in ("T", "C"):
                a, o = qA[d][f"{m} (k={k})"], qO[d][f"{m} (k={k})"]
                row[f"{m} (A)"], row[f"{m} ({other})"], row[f"Δ{m}"] = a, o, o - a
            rows.append(row)
    return pd.DataFrame(rows)


def table_global(qA, qO, other, with_alpha):
    rows = []
    for d in D_LIST:
        row = {"d": d}
        for key in ("rho", "Stress-1"):
            a, o = qA[d][key], qO[d][key]
            row[f"{key} (A)"], row[f"{key} ({other})"], row[f"Δ{key}"] = a, o, o - a
        if with_alpha:
            row["α (A)"], row[f"α ({other})"] = qA[d]["alpha"], qO[d]["alpha"]
        rows.append(row)
    return pd.DataFrame(rows)


def print_table(df, title):
    print(f"\n=== {title} ===")
    print(df.round(3).to_string(index=False))


# ----------------------------------------------------------------------------
# Bendru objektu radimas
# ----------------------------------------------------------------------------
def match_rows(X_A, y_A, X_C, y_C):
    """Kiekvienai C eilutei grazina jos eilutes numeri A aibeje (tvarka kaip C)."""
    def key(x, lab):
        return tuple(np.round(x, 9)) + (str(lab),)

    pool = {}
    for i, (x, lab) in enumerate(zip(X_A, y_A)):
        pool.setdefault(key(x, lab), []).append(i)

    idx = []
    for x, lab in zip(X_C, y_C):
        positions = pool.get(key(x, lab))
        if not positions:
            raise ValueError("C aibes eilute nerasta A aibeje – patikrink failus.")
        idx.append(positions.pop(0))
    return np.array(idx)


# ----------------------------------------------------------------------------
# 7.1 A -> B
# ----------------------------------------------------------------------------
def section_7_1(X_A, y_A, proj_A):
    print("\n" + "=" * 70 + "\n7.1 A -> B (objektai tie patys)\n" + "=" * 70)
    df_B = pd.read_csv(CSV_B)
    if len(df_B) != len(y_A) or not (df_B[CLASS_COL].to_numpy() == y_A.to_numpy()).all():
        raise ValueError("B aibes objektai ar ju tvarka nesutampa su A.")
    X_B = StandardScaler().fit_transform(df_B.drop(columns=CLASS_COL).to_numpy(float))
    print(f"A: {X_A.shape[1]} pozymiu, B: {X_B.shape[1]} pozymiu, objektu: {len(df_B)}")

    proj_B = projections_other(X_B, CSV_B.stem, "B")

    # Kiekviena projekcija vertinama savo pradineje erdveje
    qA = evaluate(X_A, proj_A, K_LIST, scale=False)
    qB = evaluate(X_B, proj_B, K_LIST, scale=False)

    print_table(table_T_C(qA, qB, "B", K_LIST), "7.1 A -> B: T ir C")
    print_table(table_global(qA, qB, "B", with_alpha=False),
                "7.1 A -> B: Spirmeno rho ir Stress-1")
    print("Patikrinimas: A stulpeliai turi sutapti su 6 skyriaus lentelemis.")


# ----------------------------------------------------------------------------
# 7.2 A -> C
# ----------------------------------------------------------------------------
def section_7_2(X_A, proj_A):
    print("\n" + "=" * 70 + "\n7.2 A -> C (vertinama tik A ∩ C, atskaitos erdve A)\n" + "=" * 70)
    df_A = pd.read_csv(DATA_PATH)                 # tas pats failas, kuri skaito load_data()
    df_C = pd.read_csv(CSV_C)
    features = [c for c in df_A.columns if c != CLASS_COL]
    X_C_raw = df_C[features].to_numpy(float)
    y_C = df_C[CLASS_COL].reset_index(drop=True)

    idx = match_rows(df_A[features].to_numpy(float), df_A[CLASS_COL], X_C_raw, y_C)
    removed = np.ones(len(df_A), dtype=bool)
    removed[idx] = False
    print(f"A: {len(df_A)}, C: {len(df_C)}, pasalinta: {removed.sum()} ({removed.mean():.1%})")
    print("Pasalinta pagal klase:")
    print(df_A.loc[removed, CLASS_COL].value_counts().to_string())

    # C projekcija: C standartizuojama savo parametrais
    X_C = StandardScaler().fit_transform(X_C_raw)
    proj_C = projections_other(X_C, CSV_C.stem, "C")

    # A projekcija apskaiciuota su visais A objektais; vertinimui paliekami tik bendri
    proj_A_common = {d: Y[idx] for d, Y in proj_A.items()}

    # Atskaitos erdve A: bendri objektai, standartizuoti A parametrais.
    # Projekcija ir atskaitos erdve gali buti skirtingo mastelio, todel Stress-1
    # skaiciuojamas su optimaliu mastelio koeficientu α (T, C ir rho nuo mastelio nepriklauso).
    X_ref = X_A[idx]
    qA = evaluate(X_ref, proj_A_common, [K_AC], scale=True)
    qC = evaluate(X_ref, proj_C, [K_AC], scale=True)

    print_table(table_T_C(qA, qC, "C", [K_AC]), f"7.2 A -> C: T ir C (k = {K_AC})")
    print_table(table_global(qA, qC, "C", with_alpha=False),
                "7.2 A -> C: Spirmeno rho, Stress-1")


# ----------------------------------------------------------------------------
def main():
    X_A, y_A, _ = load_data()
    y_A = pd.Series(y_A).reset_index(drop=True)
    print(f"A aibe: {X_A.shape[0]} objektu, {X_A.shape[1]} pozymiu")
    print(f"MDS nustatymai: sekla {SEED}, n_init = {N_INIT}, max_iter = {MAX_ITER}, "
          f"d = {D_LIST}\n")

    proj_A = projections_A(X_A)
    section_7_1(X_A, y_A, proj_A)
    section_7_2(X_A, proj_A)


if __name__ == "__main__":
    main()