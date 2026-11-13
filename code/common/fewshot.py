"""Few-shot evaluation used throughout the paper.

For every readout, split, shot count k and draw, logistic regression is fit on the shots, and its validation balanced
accuracy and test predictions are kept. external() scores the HAM10000 shots on the ISIC 2018 test set, at the readout
chosen on HAM10000 validation."""
import os
import time

import numpy as np

from common.backbones import last, readouts
from common.data import cpus, load_target, n_classes
from common.heads import balanced_acc, best_readout, fit_predict, fit_predict_arrays, readout
from common.paths import SCREEN_OUT, CREDIBILITY_OUT
from common.splits import KS, make_splits


def eval_job(path, b, cond, r, splits, y, nc, ks):
    """Runs in a worker process: everything it needs is passed in (module globals there are the defaults)."""
    import warnings
    warnings.filterwarnings("ignore")
    F = np.load(path)
    X, dims = readout(F, r, b)
    res = {}
    for sp, split in splits.items():
        for k in ks:
            val_ba, test_pred = [], []
            for idx in split["shots"][k]:
                pv, pt = fit_predict(X, y, idx, [split["val"], split["te"]], dims)
                val_ba.append(balanced_acc(pv, y[split["val"]], nc))
                test_pred.append(pt.astype(np.int8))
            res[f"{r}|{sp}|{k}|val_ba"] = np.array(val_ba)
            res[f"{r}|{sp}|{k}|test_pred"] = np.stack(test_pred)
    return cond, res


def evaluate(b, t, conds, y, nc, splits, d, ks):
    """Few-shot predictions of each condition in conds from d/feats_<cond>.npz, saved as d/preds_<cond>.npz with each
    split's test indices; finished conditions are skipped."""
    from joblib import Parallel, delayed
    conds = [c for c in conds if not (d / f"preds_{c}.npz").exists()]
    jobs = [(c, r) for c in conds for r in readouts(b)]
    print(f"{b} {t}: {len(jobs)} jobs on {cpus()} cpus", flush=True)
    t0 = time.time()
    out = Parallel(n_jobs=cpus(), verbose=2)(delayed(eval_job)(d / f"feats_{c}.npz", b, c, r, splits, y, nc, ks)
                                             for c, r in jobs)
    per = {}
    for c, res in out:
        per.setdefault(c, {}).update(res)
    for c, res in per.items():
        for sp in splits:
            res[f"split|{sp}|te"] = splits[sp]["te"]
        tmp = d / f"preds_{c}.tmp.npz"
        np.savez_compressed(tmp, **res)
        os.replace(tmp, d / f"preds_{c}.npz")
    print(f"{b} {t}: evaluated {list(per)} in {time.time() - t0:.0f}s", flush=True)


def evaluate_target(b, t, conds, ks=KS):
    """evaluate() on one of the screen's targets, with its splits, into the screen's folder."""
    tab, _ = load_target(t)
    y, g, nc = tab["y"].values, tab["group"].values, n_classes(t)
    evaluate(b, t, conds, y, nc, make_splits(y, g, nc, ks=ks), SCREEN_OUT / b / t, ks)


def fit_predict_ext(Xs, ys, Xe, dims):
    return fit_predict_arrays(Xs, ys, [Xe], dims)[0].astype(np.int8)


def ext_job(ham_path, ext_path, b, cond, splits, y, ks, chosen):
    """For each split, k and draw: fit on the HAM10000 shots at the readout chosen on HAM10000 validation, and at
    the last stage; predict the ISIC 2018 test images."""
    import warnings
    warnings.filterwarnings("ignore")
    Fh, Fe = np.load(ham_path), np.load(ext_path)
    cache_x = {}
    res = {}
    for sp, split in splits.items():
        for k in ks:
            for mode in ["best", last(b)]:
                r = chosen[sp, k] if mode == "best" else mode
                if r not in cache_x:
                    Xh, dims = readout(Fh, r, b)
                    Xe, _ = readout(Fe, r, b)
                    cache_x[r] = (Xh, Xe, dims)
                Xh, Xe, dims = cache_x[r]
                res[f"{mode}|{sp}|{k}"] = np.stack([fit_predict_ext(Xh[idx], y[idx], Xe, dims)
                                                    for idx in split["shots"][k]])
    return cond, res


def external(b, conds, ham_dir, ext_dir, splits, y, ks):
    """The ISIC 2018 test predictions (ext_dir/ext_<cond>.npz) of each condition with HAM10000 predictions and ISIC
    features; finished conditions are skipped."""
    from joblib import Parallel, delayed
    conds = [c for c in conds if (ham_dir / f"preds_{c}.npz").exists()
             and (ext_dir / f"feats_{c}.npz").exists() and not (ext_dir / f"ext_{c}.npz").exists()]

    def chosen_for(c):
        S = np.load(ham_dir / f"preds_{c}.npz")
        return {(sp, k): best_readout(S, readouts(b), sp, k) for sp in splits for k in ks}

    print(f"{b}: external test for {len(conds)} conditions", flush=True)
    out = Parallel(n_jobs=cpus(), verbose=2)(
        delayed(ext_job)(ham_dir / f"feats_{c}.npz", ext_dir / f"feats_{c}.npz", b, c, splits, y, ks,
                         chosen_for(c)) for c in conds)
    for c, res in out:
        np.savez_compressed(ext_dir / f"ext_{c}.npz", **res)


def external_ham(b, conds, ks=KS):
    """external() from HAM10000's shots to the ISIC 2018 test set, in the pipeline's folders."""
    tab, _ = load_target("ham")
    y, g = tab["y"].values, tab["group"].values
    external(b, conds, SCREEN_OUT / b / "ham", CREDIBILITY_OUT / b / "isic_test", make_splits(y, g, 7, ks=ks), y, ks)
