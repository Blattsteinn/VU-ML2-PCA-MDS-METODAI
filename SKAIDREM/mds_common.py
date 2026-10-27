"""
Bendri MDS nustatymai ir funkcijos.
"""
import inspect
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS, trustworthiness

DATA_PATH = "A24_cleaned.csv"
CLASS_COL = "class"
SEED      = 42
N_INIT    = 2
MAX_ITER  = 1000
K_LIST    = [5, 15, 50]
BASE_DIR   = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
CSV_DIR    = BASE_DIR / "csv"
CACHE_DIR  = BASE_DIR / "mds_cache"

IMAGES_DIR.mkdir(exist_ok=True)
CSV_DIR.mkdir(exist_ok=True)

_PARAMS = inspect.signature(MDS).parameters


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = StandardScaler().fit_transform(df.drop(columns=CLASS_COL).to_numpy(float))
    return X, df[CLASS_COL], pdist(X)


def make_mds(d, metric=True, n_init=N_INIT, seed=SEED):
    """MDS (SMACOF), suderinamas su senesnemis ir naujesnemis sklearn versijomis.
    n_jobs = 1 visada: lygiagretus skaiciavimas gali duoti sioki toki sprendini."""
    kw = dict(n_components=d, n_init=n_init, max_iter=MAX_ITER,
              random_state=seed, n_jobs=1)
    if "metric_mds" in _PARAMS:
        kw["metric_mds"] = metric
    else:
        kw["metric"] = metric
    if "init" in _PARAMS:
        kw["init"] = "random"
    return MDS(**kw)


def metric_mds(X, d):
    CACHE_DIR.mkdir(exist_ok=True)
    f = CACHE_DIR / (f"{Path(DATA_PATH).stem}_metric_d{d}_seed{SEED}"
                     f"_ninit{N_INIT}_iter{MAX_ITER}.npz")
    if f.exists():
        z = np.load(f)
        return z["Y"], int(z["n_iter"])
    t = time.time()
    m = make_mds(d)
    Y = m.fit_transform(X)
    np.savez(f, Y=Y, n_iter=m.n_iter_)
    print(f"  [apskaiciuota MDS d={d}: {time.time() - t:.1f} s, iter={m.n_iter_}]")
    return Y, m.n_iter_


def run_seed(X, d, seed):
    """Vienas metrinio MDS paleidimas su atsitiktine inicializacija (stabilumui)."""
    m = make_mds(d, n_init=1, seed=seed)
    Y = m.fit_transform(X)
    return Y, m.n_iter_


def stress1(dX, dY, scale=False):
    """Kruskal Stress-1. scale=True: su optimaliu mastelio koeficientu alpha
    (reikalinga nemetriniam MDS, kurio mastelis laisvas). Grazina (Stress-1, alpha)."""
    alpha = (dX @ dY) / (dY @ dY) if scale else 1.0
    return np.sqrt(((dX - alpha * dY) ** 2).sum() / (dX ** 2).sum()), alpha


def quality(X, Y, dX, k_list=K_LIST, scale=False):
    """T, C ir |T − C| kiekvienam k, Spirmeno rho ir Stress-1."""
    dY = pdist(Y)
    out = {}
    for k in k_list:
        T = trustworthiness(X, Y, n_neighbors=k)
        C = trustworthiness(Y, X, n_neighbors=k)
        out[f"T (k={k})"], out[f"C (k={k})"], out[f"|T-C| (k={k})"] = T, C, abs(T - C)
    out["rho"] = spearmanr(dX, dY)[0]
    out["Stress-1"], alpha = stress1(dX, dY, scale)
    if scale:
        out["alpha"] = alpha
    return out