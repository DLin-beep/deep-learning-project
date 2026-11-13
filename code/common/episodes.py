"""BSCD-FSL episodes (Guo et al. 2020).

An episode takes 5 of a dataset's classes, then shot + 15 query images of each, drawn without replacement and without
grouping, as in the benchmark. The episodes are drawn once, with a fixed seed per dataset and shot count. ISIC here is
HAM10000 (the ISIC 2018 Task 3 training images), and ChestX is the screen's ChestX-ray14 cache."""
import collections

import numpy as np

from common.backbones import readouts
from common.data import load_target, n_classes
from common.heads import DCBase, best_readout, dc_predict, inductive, readout

SETS, SHOTS, N_WAY, N_QUERY, N_EPISODES = ["ham", "cxr"], [1, 5], 5, 15, 600
HEADS = ["lr", "ncm", "lda", "jsncm", "jslda", "cache", "dc"]


def episodes_of(t, shot, n_episodes=N_EPISODES):
    """[(classes, support, queries)] for dataset t."""
    y = load_target(t)[0]["y"].values
    rng = np.random.RandomState(4000 + 10 * SETS.index(t) + shot)
    out = []
    for _ in range(n_episodes):
        cls = rng.choice(n_classes(t), N_WAY, replace=False)
        picks = [rng.choice(np.flatnonzero(y == c), shot + N_QUERY, replace=False) for c in cls]
        out.append((cls, np.concatenate([p[:shot] for p in picks]), np.concatenate([p[shot:] for p in picks])))
    return out


def episode_readout(preds_path, b, splits):
    """The readout the paper's LR chose most often on validation at k = 5 (ties: the earlier in readouts(b))."""
    S = np.load(preds_path)
    picks = [best_readout(S, readouts(b), sp, 5) for sp in splits]
    return max(readouts(b), key=picks.count)


def episode_job(b, t, c, shot, feats_path, preds_path, refit_path, base_path, splits, n_episodes=N_EPISODES):
    """Accuracy per episode of every inductive head, on every episode of one shot count, for one network (runs in a
    worker process). The cache term uses the (a, beta) chosen most often on validation at k = 5 at this readout, from
    the classifier check's refit file. Distribution calibration uses the base set's features."""
    import warnings
    warnings.filterwarnings("ignore")
    y = load_target(t)[0]["y"].values
    r = episode_readout(preds_path, b, splits)
    X, dims = readout(np.load(feats_path), r, b)
    Z = np.load(refit_path)
    abs_ = [tuple(Z[f"cache|{r}|{sp}|5|ab"]) for sp in splits]
    ab = max(sorted(set(abs_)), key=abs_.count)
    Fb = np.load(base_path)
    Xb = readout(Fb, r, b)[0]
    base_ = DCBase(Xb, Fb["y"], bool((Xb >= 0).all() and (X >= 0).all()))
    acc = collections.defaultdict(list)
    for e, (cls, sup, qry) in enumerate(episodes_of(t, shot, n_episodes)):
        lab = {int(v): i for i, v in enumerate(cls)}
        ys, yq = np.array([lab[int(v)] for v in y[sup]]), np.array([lab[int(v)] for v in y[qry]])
        o = inductive(X[sup], ys, [X[qry]], N_WAY, shot, dims, ab=ab)
        o["dc"] = dc_predict(base_, X[sup], ys, [X[qry]], N_WAY, shot, dims, np.random.default_rng([2027, shot, e]))
        for n in HEADS:
            if n in o:
                acc[n].append(float((o[n][0] == yq).mean()))
    return c, shot, r, ab, {f"{n}|k{shot}": np.array(v) for n, v in acc.items()}
