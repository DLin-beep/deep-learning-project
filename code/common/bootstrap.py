"""The hierarchical bootstrap behind every interval, and gains and levels computed from correctness arrays.

Each of the N_BOOT resamples (drawn from RandomState(0)) resamples the training seeds, the splits and each split's test
groups. The seed and split effects are inflated by sqrt(n/(n-1))."""
import numpy as np
import pandas as pd

from common.data import load_isic
from common.paths import ROOT

N_BOOT = 2000
LESIONS = ROOT / "pilot" / "out_s40" / "isic_lesions.csv"     # ISIC Archive lesion IDs of the ISIC 2018 images


class Boot:
    """yte, gidx: {split: test labels, test group index}. seeds: the training seeds, or None for a comparison without
    seeds (then only splits and test groups are resampled). splits: their order (default: yte's). Each option matches
    one of the older bootstrap variants:
      shared  every split scores the same test set (ISIC), so one resample of its groups serves all splits (Boot40)
      guard   a single seed or split is not inflated (Boot40); without it sqrt(n/(n-1)) divides by zero, as in S1.Boot
      nan     "ignore": classes missing from a resample are left out of its balanced accuracy (nanmean; S1.Boot);
              "propagate": plain mean, as pilot2's bootstrap"""

    def __init__(self, yte, gidx, nc, seeds=None, splits=None, n_boot=N_BOOT, shared=False, guard=False,
                 nan="ignore"):
        rng = np.random.RandomState(0)
        self.seeds = [None] if seeds is None else list(seeds)
        self.splits = list(yte) if splits is None else list(splits)
        nt, ns = len(self.seeds), len(self.splits)
        self.yte, self.nc, self.n_boot, self.guard, self.nan = yte, nc, n_boot, guard, nan
        boot_seeds = rng.randint(0, nt, size=(n_boot, nt))
        boot_splits = rng.randint(0, ns, size=(n_boot, ns))
        self.w = {}
        for i, sp in enumerate(self.splits):
            if shared and i:
                self.w[sp] = self.w[self.splits[0]]
                continue
            ng = gidx[sp].max() + 1
            draws = rng.randint(0, ng, size=(n_boot, ng))
            self.w[sp] = np.stack([np.bincount(dr, minlength=ng)[gidx[sp]] for dr in draws]).astype(float)
        self.Yoh = {sp: np.eye(nc)[yte[sp]] for sp in self.splits}
        self.den = {sp: np.einsum("in,nc->ic", self.w[sp], self.Yoh[sp]) for sp in self.splits}
        self.Ct = np.stack([np.bincount(r, minlength=nt) for r in boot_seeds]).astype(float)
        self.Cs = np.stack([np.bincount(r, minlength=ns) for r in boot_splits]).astype(float)

    def ba(self, cbar, sp):
        yy = self.yte[sp]
        return np.mean([cbar[yy == c].mean() for c in range(self.nc)])

    def dist(self, diff):
        """diff(seed, split) -> per-test-image difference in mean correctness. Returns the mean over seed x split cells
        of the balanced-accuracy difference, its bootstrap distribution, and the cells."""
        nt, ns = len(self.seeds), len(self.splits)
        G = np.empty((self.n_boot, nt, ns))
        cells = []
        for ti, t in enumerate(self.seeds):
            for si, sp in enumerate(self.splits):
                d = diff(t, sp)
                cells.append(self.ba(d, sp))
                if self.nan == "ignore":
                    with np.errstate(invalid="ignore", divide="ignore"):
                        G[:, ti, si] = np.nanmean(np.einsum("in,nc->ic", self.w[sp] * d, self.Yoh[sp]) / self.den[sp], 1)
                else:
                    G[:, ti, si] = (np.einsum("in,nc->ic", self.w[sp] * d, self.Yoh[sp]) / self.den[sp]).mean(1)
        gm = G.mean((1, 2), keepdims=True)
        rt, cs = G.mean(2, keepdims=True) - gm, G.mean(1, keepdims=True) - gm
        if nt > 1 or not self.guard:
            G = G + (np.sqrt(nt / (nt - 1)) - 1) * rt
        if ns > 1 or not self.guard:
            G = G + (np.sqrt(ns / (ns - 1)) - 1) * cs
        return float(np.mean(cells)), np.einsum("it,its,is->i", self.Ct, G, self.Cs) / (nt * ns), cells

    def gain(self, diff):
        """The mean gain, its 95% interval, and the seed x split cells."""
        g, bs, cells = self.dist(diff)
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        return g, float(lo), float(hi), cells


def boot_ext(n_test, y_ext, seeds, splits):
    """The paper's bootstrap for the ISIC 2018 test set. Each shot set (split) scores all n_test images, which are
    resampled separately for each split, each image its own group."""
    return Boot({sp: y_ext for sp in splits}, {sp: np.arange(n_test) for sp in splits}, 7, seeds, splits)


def isic_units(lesions=LESIONS):
    """Resampling units of the ISIC 2018 test images. These are lesions once the archive's lesion IDs have been fetched
    (an image without one is its own unit), and the images themselves otherwise. Returns (unit index per image,
    description)."""
    ids = load_isic("isic_test")[0]["group"].astype(str).values          # the image ID (isic_table)
    if lesions.exists():
        m = pd.read_csv(lesions)
        m = m[m.part == "test"]
        lid = {i: l for i, l in zip(m.isic_id, m.lesion_id) if isinstance(l, str)}
        u = pd.factorize(np.array([lid.get(i, i) for i in ids]))[0]
        return u, f"lesions ({u.max() + 1} units; {sum(i not in lid for i in ids)} images without a lesion ID)"
    return np.arange(len(ids)), f"images ({len(ids)}; lesion IDs not fetched)"


# ---------------------------------------------------------------- gains and levels from correctness arrays
def _get(C, x, t, sp, k):
    b, f = x
    return C[b, f if f == "imagenet" else f"{f}_s{t}"][sp, k]


def compare(Bx, C, a, b, ks):
    """a minus b, from C[(backbone, condition)][(split, k)] = draws x test images correctness, averaged over ks."""
    gg, lo, hi, cells = Bx.gain(lambda t, sp: np.mean([_get(C, a, t, sp, k).mean(0) - _get(C, b, t, sp, k).mean(0)
                                                       for k in ks], 0))
    return dict(gain=100 * gg, lo=100 * lo, hi=100 * hi, cells=f"{sum(x > 0 for x in cells)}/{len(cells)}")


def level(C, x, yte, nc, ks, seeds, splits):
    """Balanced accuracy (%), mean over draws, splits, k and seeds."""
    seeds = [0] if x[1] == "imagenet" else seeds
    return 100 * np.mean([np.mean([_get(C, x, t, sp, k)[:, yte[sp] == c].mean() for c in range(nc)])
                          for t in seeds for sp in splits for k in ks])


# ---------------------------------------------------------------- p-values
def boot_p(bs):
    """Two-sided bootstrap p-value (the smallest alpha at which the percentile interval excludes 0)."""
    return float(min(1.0, 2 * max(min((bs <= 0).mean(), (bs >= 0).mean()), 0.5 / len(bs))))


def holm(p):
    p = np.asarray(p, float)
    o, m = np.argsort(p), len(p)
    adj = np.empty(m)
    adj[o] = np.minimum(np.maximum.accumulate((m - np.arange(m)) * p[o]), 1)
    return adj


def bh(p):
    p = np.asarray(p, float)
    o, m = np.argsort(p), len(p)
    adj = np.empty(m)
    adj[o] = np.minimum(np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1], 1)
    return adj
