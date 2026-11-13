"""The linear-map test.

A ridge map from one network's features to another's is fit without labels. The paper's few-shot protocol then runs
on in-memory feature sets (all layers concatenated and standardized), for HAM10000 (whose shots also score the ISIC
2018 test set) and the internal skin tests."""
import numpy as np
import pandas as pd

from common.backbones import is_vit, stages
from common.data import cpus, load_isic, load_target, target_dir
from common.heads import balanced_acc, fit_predict_arrays
from common.paths import OUT_S4
from common.splits import KS, OTHER_KS, SPLITS, make_splits

INTERNAL = ["pad_ufes20", "hiba_derm", "bcn20000"]
RIDGE = np.logspace(-2, 4, 13)


# ---------------------------------------------------------------- features
def layers(b):
    return stages(b) + (["cls"] if is_vit(b) else [])


def load(b, t, cond):
    """A network's readouts on a target, concatenated (all-layer), as float32."""
    F = np.load(target_dir(b, t) / f"feats_{cond}.npz")
    return np.hstack([F[r].astype(np.float32) for r in layers(b)])


def blocks(Z, b):
    d = Z.shape[1] // len(layers(b))
    return {r: np.ascontiguousarray(Z[:, i * d:(i + 1) * d]) for i, r in enumerate(layers(b))}


class Std:
    """Per-feature standardization with the statistics of the images it is built on."""
    def __init__(self, X):
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd < 1e-8] = 1.0

    def __call__(self, X):
        return ((X - self.mu) / self.sd).astype(np.float32)

    def inverse(self, Z):
        return (Z * self.sd + self.mu).astype(np.float32)


# ---------------------------------------------------------------- the paper's few-shot protocol on in-memory features
def candidates(Z, b, extra_all=False):
    """The paper's candidate readouts (each stage, their concatenation, the class token) from the all-layer features,
    optionally with all layers together."""
    L = blocks(Z, b)
    st = stages(b)
    R = {r: (L[r], None) for r in st}
    R["cat"] = (np.hstack([L[r] for r in st]), [L[r].shape[1] for r in st])
    R["cls"] = (L["cls"], None)
    if extra_all:
        R["all"] = (np.ascontiguousarray(Z), [L[r].shape[1] for r in layers(b)])
    return R


def fit_split(Z, b, y, split, ks, nc, Ze=None, extra_all=False, draws=None):
    """One split, as in the pipeline. Every readout is fit per k on the first `draws` shot draws (all when None). The
    readout with the best mean validation balanced accuracy (the first, on ties) gives the test predictions, and the
    ISIC ones when Ze holds ISIC's features."""
    import warnings
    warnings.filterwarnings("ignore")
    R = candidates(Z, b, extra_all)
    Re = candidates(Ze, b, extra_all) if Ze is not None else None
    out = {}
    for k in ks:
        best = None
        for r, (X, dims) in R.items():
            val, te, ex = [], [], []
            for idx in split["shots"][k][:draws]:
                p = fit_predict_arrays(X[idx], y[idx], [X[split["val"]], X[split["te"]]] + ([Re[r][0]] if Re else []),
                                       dims)
                val.append(balanced_acc(p[0], y[split["val"]], nc))
                te.append(p[1])
                ex += p[2:]
            if best is None or np.mean(val) > best[0]:
                best = (np.mean(val), r, np.stack(te).astype(np.int8), np.stack(ex).astype(np.int8) if ex else None)
        out[k] = best[1:]
    return out


def skin_tests(internal=INTERNAL, other_ks=OTHER_KS):
    """Labels and splits of HAM10000 (k = 5/20/50, its shots also tested on ISIC) and the internal skin tests, and the
    ISIC 2018 test labels."""
    tab, _ = load_target("ham")
    T = {"ham": dict(y=tab.y.values, g=tab.group.values, nc=7, ks=list(KS),
                     splits=make_splits(tab.y.values, tab.group.values, 7, ks=list(KS)))}
    for t in internal:
        tb = pd.read_csv(OUT_S4 / "cache" / f"{t}_index.csv")
        nc = int(tb.y.max()) + 1
        T[t] = dict(y=tb.y.values, g=tb.group.values, nc=nc, ks=list(other_ks),
                    splits=make_splits(tb.y.values, tb.group.values, nc, ks=list(other_ks)))
    return T, load_isic("isic_test")[0]["y"].values


def evaluate(sets, b, T, ye, extra_all=(), splits=SPLITS, internal=INTERNAL, draws=None):
    """Few-shot correctness of named all-layer feature sets ({name: {test: array}}), flat-keyed test|name|split|k,
    and the readout chosen for each."""
    from joblib import Parallel, delayed
    jobs = [(n, t, sp) for n, Zs in sets.items() for t in ["ham"] + internal if t in Zs for sp in splits]
    res = Parallel(n_jobs=cpus(), verbose=1)(
        delayed(fit_split)(sets[n][t], b, T[t]["y"], T[t]["splits"][sp], T[t]["ks"], T[t]["nc"],
                           sets[n].get("isic_test") if t == "ham" else None, n in extra_all, draws)
        for n, t, sp in jobs)
    C, chosen = {}, {}
    for (n, t, sp), out in zip(jobs, res):
        yt = T[t]["y"][T[t]["splits"][sp]["te"]]
        for k, (r, te, ex) in out.items():
            C[f"{t}|{n}|{sp}|{k}"] = te == yt[None]
            chosen[f"{t}|{n}|{sp}|{k}"] = r
            if ex is not None:
                C[f"isic_test|{n}|{sp}|{k}"] = ex == ye[None]
    return C, chosen


# ---------------------------------------------------------------- the maps
def ridge(X, Y, alpha=None):
    """Ridge map Y ~ X A + c between standardized feature sets (strength by leave-one-out unless given)."""
    from sklearn.linear_model import Ridge, RidgeCV
    m = (Ridge(alpha=alpha) if alpha else RidgeCV(alphas=RIDGE)).fit(X, Y)
    return m.coef_.T.astype(np.float32), m.intercept_.astype(np.float32), float(alpha or m.alpha_)


def r2(Y, Yhat):
    return float(1 - ((Y - Yhat) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def spectrum(A, X):
    """The stand-in's amplification spectrum: the eigenvalues of B B', B the map after the classifier's per-feature
    standardization, scaled to mean 1 (descending, floored at 1e-3)."""
    sd = (X @ A).std(0)
    sd[sd < 1e-8] = 1.0
    Bm = (A / sd).astype(np.float64)
    G = Bm @ Bm.T
    G *= len(G) / np.trace(G)
    return np.maximum(np.linalg.eigvalsh(G)[::-1], 1e-3)
