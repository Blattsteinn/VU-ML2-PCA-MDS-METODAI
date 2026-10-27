import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

CSV = ["A24_no_redundant.csv"]

LABEL = "class"
DROP_COLS = []

for file in CSV:
    df = pd.read_csv(file)
    X = df.drop(columns=[LABEL] + DROP_COLS)

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA().fit(X_scaled)

    weights = pd.DataFrame(pca.components_[:2].T, index=X.columns, columns=["PK1", "PK2"])

    print(f"\n=== {file} ===")
    print("\nSorted by |PK1|:")
    print(weights.sort_values("PK1", key=abs, ascending=False).round(3).to_string())
    print("\nSorted by |PK2|:")
    print(weights.sort_values("PK2", key=abs, ascending=False).round(3).to_string())
