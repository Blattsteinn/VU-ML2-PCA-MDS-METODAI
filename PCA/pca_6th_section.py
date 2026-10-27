"""

Projekciju kokybes ir stabilumo vertinimas (PCA)

"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

CSV_PATH = "A24_cleaned.csv"

CLASS_COL = "class"
CSV_SEP = ","

K_LIST = [5, 15, 50]
D_LIST = [2, 3]
N_BOOT = 100
N_PC_STAB = 5
KNN_CLF_K = 5
SEED = 42


def stress_1(X, Y):
    dX, dY = pdist(X), pdist(Y)
    return float(np.sqrt(np.sum((dX - dY) ** 2) / np.sum(dX ** 2)))



def quality_tables(X_ref, labels, d_list=D_LIST, k_list=K_LIST):
    global_rows, local_rows, class_rows = [], [], []
    for d in d_list:
        pca = PCA(n_components=d, svd_solver="full", random_state=SEED)
        Y = pca.fit_transform(X_ref)

        global_rows.append(
            {
                "d": d,
                "Dispersija": float(pca.explained_variance_ratio_.sum()),
                "Spearman rho": float(spearmanr(pdist(X_ref), pdist(Y))[0]),
                "Stress-1": stress_1(X_ref, Y),
            }
        )

        row = {"d": d}
        for k in k_list:
            row[f"T (k={k})"] = trustworthiness(X_ref, Y, n_neighbors=k)
            row[f"C (k={k})"] = trustworthiness(Y, X_ref, n_neighbors=k)
        local_rows.append(row)


    return pd.DataFrame(global_rows), pd.DataFrame(local_rows), pd.DataFrame(class_rows)


def main():
    df = pd.read_csv(CSV_PATH, sep=CSV_SEP)
    labels = df[CLASS_COL].to_numpy()
    X_raw = df.drop(columns=[CLASS_COL]).to_numpy(dtype=float)
    print(f"Objektu: {X_raw.shape[0]}, pozymiu: {X_raw.shape[1]}, klasiu: {len(np.unique(labels))}")

    X_ref = StandardScaler().fit_transform(X_raw)

    global_tbl, local_tbl, class_tbl = quality_tables(X_ref, labels)

    print("\n--- Globali struktura ")
    print(global_tbl.round(3).to_string(index=False))
    print("\n--- Lokali struktura (T, C)")
    print(local_tbl.round(3).to_string(index=False))
    baseline = pd.Series(labels).value_counts(normalize=True).max()
    print(f"Atskaitos lygis (didziausios klases dalis): {baseline:.3f}")

if __name__ == "__main__":
    main()
