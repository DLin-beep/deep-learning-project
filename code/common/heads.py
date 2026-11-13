"""Few-shot heads and the readout rule.

Every head sees features standardized with the shots' mean and SD. In the concatenated readout (cat), each stage is
weighted to the same total variance.
  lr      logistic regression, C = 0.1 (the paper's head)
  ncm     nearest class mean (Euclidean); cosine_ncm: on L2-normalized features (the robustness check)
  lda     sklearn's LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), solved exactly in low rank
  jsncm   nearest class mean and LDA with the class means shrunk toward their grand mean (Efron-Morris, per feature)
  jslda
  cache   Tip-Adapter's training-free cache term added to LR's logits
  dc      distribution calibration (Yang et al. 2021)
  atim    alpha-TIM (Veilleux et al. 2021), transductive
"""
import numpy as np
import torch

from common.backbones import stages

C_LR = 0.1                                       # logistic regression strength, fixed, not tuned
CACHE_A, CACHE_BETA = [0.5, 1, 2, 4, 8], [1, 3, 5.5, 10]
DC_TOP, DC_ALPHA, DC_PER_CLASS = 2, 0.21, 750
ATIM_ITERS, ATIM_LR, ATIM_TEMP = 1000, 1e-4, 15.0


# ---------------------------------------------------------------- readouts
def readout(F, r, b):
    """Readout r of a feature file as float32, and, for the concatenation of the stages (cat), each stage's width."""
    if r != "cat":
        return F[r].astype(np.float32), None
    blocks = [F[s].astype(np.float32) for s in stages(b)]
    return np.hstack(blocks), [x.shape[1] for x in blocks]


def best_readout(S, rs, sp, k, prefix="", draws=None):
    """The readout of rs with the highest mean validation balanced accuracy over the draws (the first on ties), from
    S[f"{prefix}{r}|{sp}|{k}|val_ba"]."""
    return max(rs, key=lambda r: S[f"{prefix}{r}|{sp}|{k}|val_ba"][:draws].mean())


# ---------------------------------------------------------------- logistic regression
def shot_scaling(Xs, dims):
    """The shots' mean and SD (an SD under 1e-8 counts as 1), and the stage weights of the concatenated readout."""
    mu, sd = Xs.mean(0), Xs.std(0)
    sd[sd < 1e-8] = 1.0
    scale = 1.0 if dims is None else np.concatenate([np.full(d, np.sqrt(sum(dims) / (len(dims) * d))) for d in dims])
    return mu, sd, scale


def standardizer(Xs, dims):
    mu, sd, scale = shot_scaling(Xs, dims)
    return lambda A: (A - mu) / sd * scale


def lr_fit(Z, y, C=C_LR):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, max_iter=3000).fit(Z, y)


def fit_predict_arrays(Xs, ys, evals, dims, C=C_LR, proba=False):
    """Logistic regression on the shots (Xs, ys); predictions (or probabilities) for each array in evals."""
    f = standardizer(Xs, dims)
    m = lr_fit(f(Xs), ys, C)
    g = m.predict_proba if proba else m.predict
    return [g(f(E)) for E in evals]


def fit_predict(X, y, idx, evals, dims, C=C_LR):
    """Logistic regression on the shots X[idx]; predictions for each index array in evals."""
    return fit_predict_arrays(X[idx], y[idx], [X[e] for e in evals], dims, C)


def balanced_acc(pred, y, ncls=7):
    return np.mean([(pred[y == c] == c).mean() for c in range(ncls)])


# ---------------------------------------------------------------- the other heads
def nearest(Z, M):
    return (Z @ M.T - 0.5 * (M ** 2).sum(1)).argmax(1)


def unit(A, eps=1e-12):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)


def cosine_ncm(X, y, idx, evals, dims, nc):
    """Nearest class mean on L2-normalized standardized features, as in SimpleShot (the robustness check)."""
    z = standardizer(X[idx], dims)
    Zs = unit(z(X[idx]), 1e-8)
    M = unit(np.stack([Zs[y[idx] == c].mean(0) for c in range(nc)]), 1e-8)
    return [np.argmax(unit(z(X[e]), 1e-8) @ M.T, 1) for e in evals]


def lda_solver(Z, y, nc):
    """sklearn's LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), solved exactly for any class means. sklearn
    shrinks each class covariance with Ledoit-Wolf in that class's standardized coordinates and averages them with the
    class priors, so the covariance is the centred shots times their transpose plus a diagonal, and the Woodbury identity
    needs an n x n solve instead of a d x d one. Returns means -> (coef, intercept)."""
    from scipy import linalg
    from sklearn.covariance import ledoit_wolf_shrinkage
    from sklearn.preprocessing import StandardScaler
    Z = np.asarray(Z, np.float64)
    n, d = Z.shape
    priors = np.bincount(y, minlength=nc) / n
    rows, diag = [], np.zeros(d)
    for c in range(nc):
        Xc = Z[y == c]
        sc = StandardScaler().fit(Xc)
        Zc = sc.transform(Xc)
        delta = ledoit_wolf_shrinkage(Zc)
        mu = ((Zc - Zc.mean(0)) ** 2).sum() / (len(Xc) * d)            # trace of the empirical covariance / d
        rows.append(np.sqrt(priors[c] * (1 - delta) / len(Xc)) * (Xc - sc.mean_))
        diag += priors[c] * delta * mu * sc.scale_ ** 2
    U = np.vstack(rows).T                                               # d x n
    if diag.min() <= 0:                                                 # no shrinkage at all: sklearn's dense path
        cov = U @ U.T + np.diag(diag)

        def solve(M):
            coef = linalg.lstsq(cov, M.T)[0].T
            return coef, -0.5 * np.einsum("cd,cd->c", M, coef) + np.log(priors)
        return solve
    UL = U / diag[:, None]
    K = np.eye(U.shape[1]) + U.T @ UL

    def solve(M):
        B = M.T / diag[:, None]
        coef = (B - UL @ np.linalg.solve(K, U.T @ B)).T
        return coef, -0.5 * np.einsum("cd,cd->c", M, coef) + np.log(priors)
    return solve


def js_means(Z, y, nc, k):
    """Efron-Morris positive-part James-Stein, per feature: the C class means shrunk toward their grand mean by
    B_j = min(1, (C - 3)/(nu + 2) * W_j / (k * sum_c (m_cj - mbar_j)^2)), with W_j the within-class sum of squares and
    nu = C(k - 1) its degrees of freedom. Needs C >= 4 and k >= 2. Returns the shrunk means and B."""
    M = np.stack([Z[y == c].mean(0) for c in range(nc)])
    W = sum(((Z[y == c] - M[c]) ** 2).sum(0) for c in range(nc))
    mbar = M.mean(0)
    S = ((M - mbar) ** 2).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        B = np.where(S > 0, np.minimum(1.0, (nc - 3) / (nc * (k - 1) + 2) * W / (k * S)), 1.0)
    return mbar + (1 - B) * (M - mbar), B


def inductive(Xs, ys, evals, nc, k, dims, ab=None):
    """LR, NCM, LDA, JS-NCM, JS-LDA and the cache term on one set of shots, predicting each array in evals. The cache term
    is computed at (a, beta) = ab, or, when ab is None, over the whole grid ({(a, beta): predictions}). LDA and the
    James-Stein classifiers need at least 2 shots per class and are left out at 1."""
    f = standardizer(Xs, dims)
    Zs = np.asarray(f(Xs), np.float64)
    Ze = [np.asarray(f(E), np.float64) for E in evals]
    m = lr_fit(Zs, ys)
    L = [m.decision_function(Z) for Z in Ze]
    M = np.stack([Zs[ys == c].mean(0) for c in range(nc)])
    out = {"lr": [l.argmax(1) for l in L], "ncm": [nearest(Z, M) for Z in Ze]}
    if k >= 2:
        solve = lda_solver(Zs, ys, nc)
        Mj, B = js_means(Zs, ys, nc, k)
        out["jsncm"] = [nearest(Z, Mj) for Z in Ze]
        for name, means in [("lda", M), ("jslda", Mj)]:
            coef, icpt = solve(means)
            out[name] = [(Z @ coef.T + icpt).argmax(1) for Z in Ze]
        out["js_b"] = float(B.mean())
    Y = np.eye(nc)[ys] / k
    cos = [unit(Z) @ unit(Zs).T for Z in Ze]
    cache = {}
    for beta in (CACHE_BETA if ab is None else [ab[1]]):
        E = [np.exp(-beta * (1 - cs)) @ Y for cs in cos]
        for a in (CACHE_A if ab is None else [ab[0]]):
            cache[a, beta] = [(l + a * e).argmax(1) for l, e in zip(L, E)]
    out["cache"] = cache if ab is None else cache[ab[0], ab[1]]
    return out


class DCBase:
    """Distribution calibration's base statistics under one network and readout: Tukey's transform (lambda = 0.5) where
    every feature is non-negative, standardization with the base set's mean and SD, then each base class's mean and a
    square root of its covariance (its centred images / sqrt(n - 1)), so that no d x d matrix is formed."""

    def __init__(self, Xb, yb, nonneg):
        self.tukey = nonneg
        Xb = np.sqrt(np.maximum(Xb, 0)) if nonneg else Xb
        self.mu, self.sd = Xb.mean(0), Xb.std(0)
        self.sd[self.sd < 1e-8] = 1.0
        Z = (Xb - self.mu) / self.sd
        self.means = np.stack([Z[yb == j].mean(0) for j in range(int(yb.max()) + 1)])
        self.roots = [(Z[yb == j] - self.means[j]) / np.sqrt(max(int((yb == j).sum()) - 1, 1))
                      for j in range(len(self.means))]

    def tf(self, A):
        return ((np.sqrt(np.maximum(A, 0)) if self.tukey else A) - self.mu) / self.sd


def dc_draw(base, x, n, rng):
    """n features drawn from one shot's calibrated distribution. Its mean is the mean of the shot and its DC_TOP nearest
    base-class means. Its covariance is the mean of those classes' covariances with alpha added to every entry, as
    `cov + alpha` does in the original DC code."""
    top = np.argsort(((base.means - x) ** 2).sum(1))[:DC_TOP]
    mu = (base.means[top].sum(0) + x) / (DC_TOP + 1)
    noise = sum(rng.standard_normal((n, len(base.roots[j]))) @ base.roots[j] for j in top) / np.sqrt(DC_TOP)
    return mu + noise + np.sqrt(DC_ALPHA) * rng.standard_normal((n, 1))


def dc_predict(base, Xs, ys, evals, nc, k, dims, rng):
    """Distribution calibration (Yang et al. 2021): 750 / k features drawn per shot, then our LR on the shots plus the
    drawn features, in the base set's coordinates."""
    Zs, Ze = base.tf(Xs), [base.tf(E) for E in evals]
    per = DC_PER_CLASS // k
    X = np.vstack([Zs] + [dc_draw(base, x, per, rng) for x in Zs])
    Y = np.concatenate([ys] + [np.full(per, yy) for yy in ys])
    f = standardizer(X, dims)
    m = lr_fit(f(X), Y)
    return [m.predict(f(Z)) for Z in Ze]


def atim_predict(Xs, ys, Xq, dims, nc, alpha, dev, iters=ATIM_ITERS):
    """alpha-TIM (Veilleux et al. 2021) for a batch of draws, with its authors' class-imbalanced settings.

    Features are standardized as for the other classifiers, then L2-normalized. Weights start at the support's class
    means, and Adam (learning rate 1e-4) runs 1,000 steps on cross-entropy - H_alpha(Y) + H_alpha(Y|X), with loss
    weights 1, 1, 1 and Tsallis alpha-entropies. Logits are -temp/2 * ||z - w||^2. Xs: (D, ns, d), ys: (D, ns),
    Xq: (nq, d), the same queries for every draw. Returns (D, nq) predictions."""
    import torch.nn.functional as Fn
    xs = torch.as_tensor(np.asarray(Xs, np.float32), device=dev)
    mu, sd = xs.mean(1, keepdim=True), xs.std(1, unbiased=False, keepdim=True)
    sd = torch.where(sd < 1e-8, torch.ones_like(sd), sd)
    scale = 1.0 if dims is None else torch.as_tensor(np.concatenate(
        [np.full(d, np.sqrt(sum(dims) / (len(dims) * d))) for d in dims]).astype(np.float32), device=dev)
    zs = Fn.normalize((xs - mu) / sd * scale, dim=2)
    zq = Fn.normalize((torch.as_tensor(np.asarray(Xq, np.float32), device=dev)[None] - mu) / sd * scale, dim=2)
    yoh = Fn.one_hot(torch.as_tensor(np.asarray(ys), device=dev).long(), nc).float()
    w = ((yoh.transpose(1, 2) @ zs) / yoh.sum(1)[:, :, None]).clone().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=ATIM_LR)

    def logits(z):
        return ATIM_TEMP * (z @ w.transpose(1, 2) - 0.5 * (w ** 2).sum(2)[:, None, :] - 0.5 * (z ** 2).sum(2)[:, :, None])

    for _ in range(iters):
        ce = -(yoh * torch.log_softmax(logits(zs), 2)).sum(2).mean(1)
        pq = torch.softmax(logits(zq), 2)
        h_marg = (1 - (pq.mean(1) ** alpha).sum(1)) / (alpha - 1)
        h_cond = ((1 - (pq ** alpha).sum(2)) / (alpha - 1)).mean(1)
        loss = (ce - h_marg + h_cond).sum()                          # draws are independent; Adam is per element
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return logits(zq).argmax(2).cpu().numpy()
