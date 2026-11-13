"""Does plant training add new information, or reweight what the network's ImageNet features already hold?

The test replaces the plant network with a linear stand-in. The stand-in is a ridge map from the ImageNet DINO ViT-S
features to the healthy-vs-diseased plant network's, fit on HAM10000 images without labels. The tests and decision rules
were fixed before any run. Features are frozen throughout, and every derived feature set goes through the paper's
few-shot protocol, re-run here.

Steps, each run as its own SLURM job:
  extract --task i  plant and Food-101 images under the networks the maps need (CPU; i in 0-2)
  standin           the stand-in and its controls on HAM10000 -> ISIC, PAD-UFES-20, HIBA and BCN20000 (CPU)
  summarize         -> pilot/out_s37/summary.txt (CPU)
S37_QUICK=1 runs a smoke test (one seed, one split, two draws) into pilot/out_s37/quick.
"""
import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

for _d in sorted(Path(__file__).resolve().parents[1].glob("[1-9]_*")):   # the paper's nine experiments
    sys.path.insert(0, str(_d))
import e1_pilot as P        # noqa: E402
import e1_pilot2 as Q       # noqa: E402
import e1_screen as S1      # noqa: E402
import e2_credibility as S4      # noqa: E402
import e7_stronger_networks as S22    # noqa: E402
from common import linear_map as _LM  # noqa: E402
from common.heads import fit_predict_arrays as fit  # noqa: E402
from common.linear_map import (RIDGE, Std, blocks, candidates, layers, load, r2, ridge,  # noqa: E402
                               spectrum)

QUICK = os.environ.get("S37_QUICK") == "1"
OUT = P.ROOT / "pilot" / "out_s37" / ("quick" if QUICK else "")
SEEDS = S22.SEEDS[:1] if QUICK else S22.SEEDS
SPLITS = S1.SPLITS[:1] if QUICK else S1.SPLITS
B = "vits_dino"
TESTS = {"isic_test": "ISIC 2018 test", "ham": "HAM10000", "pad_ufes20": "PAD-UFES-20", "hiba_derm": "HIBA",
         "bcn20000": "BCN20000"}
INTERNAL = ["pad_ufes20", "hiba_derm", "bcn20000"]
CORE = [("plant_disease2", "imagenet"), ("standin", "imagenet"), ("standin", "imagenet_all"),
        ("imagenet_all", "imagenet"), ("standin_plantfit", "imagenet"), ("standin_foodfit", "imagenet"),
        ("food", "imagenet"), ("food_standin", "imagenet"), ("zca", "imagenet"), ("randmap", "imagenet"),
        ("mlp", "imagenet"), ("mlp", "standin")]


# ---------------------------------------------------------------- features
class Skin:
    """A network's ImageNet-only all-layer features on the skin tests, raw and standardized with HAM10000's
    statistics."""
    def __init__(self, b=B):
        self.b = b
        self.tests = [t for t in TESTS if (S4.target_dir(b, t) / "feats_imagenet.npz").exists()]
        self.raw = {t: load(b, t, "imagenet") for t in self.tests}
        self.std = Std(self.raw["ham"])
        self.X = {t: self.std(v) for t, v in self.raw.items()}

    def net(self, cond):
        """Another network's features on the same tests, standardized with its own HAM10000 statistics."""
        raw = {t: load(self.b, t, cond) for t in self.tests}
        std = Std(raw["ham"])
        return {t: std(v) for t, v in raw.items()}, std


def image_set(name, seed=0):
    """(items, labels): plant_disease2's training images of one seed (healthy 0 / diseased 1), or Food-101's source
    images of seed 0."""
    if name == "plant":
        items, _ = S1.source_items("plant_disease2", random.Random(seed))
    else:
        items, _ = S1.source_items("food", random.Random(0))
    return items, np.array([y for _, y in items])


def extract(task):
    """Features of the source images that the maps need, by task:
      0  ImageNet DINO ViT-S on every seed's plant images
      1  each seed's plant network on its own plant images
      2  ImageNet and the six plant networks on Food-101 photos
    Preprocessing is as for the targets' caches (224 px resize and centre crop)."""
    import torch
    from torchvision import transforms
    torch.set_num_threads(max(1, S1.cpus() - 4))
    jobs = ([(B, "imagenet" if task == 0 else f"plant_disease2_s{s}", "plant", s) for s in SEEDS] if task < 2 else
            [(B, c, "food", 0) for c in ["imagenet"] + [f"plant_disease2_s{s}" for s in SEEDS]])
    tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.PILToTensor()])
    for b, cond, name, seed in jobs:
        path = OUT / "aux" / b / f"{name}{seed}_{cond}.npz"
        if path.exists():
            continue
        t0 = time.time()
        items, y = image_set(name, seed)
        if QUICK:                                      # 64 images at random
            pick = sorted(random.Random(0).sample(range(len(items)), 64))
            items, y = [items[i] for i in pick], y[pick]
        mean, std = (torch.tensor(v).view(1, 3, 1, 1) for v in S1.norm_stats(b))
        m = S1.build(b, cond).eval()
        feats = {}
        with torch.no_grad():
            for x, _ in torch.utils.data.DataLoader(P.ImageList(list(items), tf), batch_size=64, num_workers=4):
                for k, v in S1.stage_features(b, m, (x.float().div(255) - mean) / std).items():
                    feats.setdefault(k, []).append(v.numpy().astype(np.float16))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path.with_suffix(".tmp.npz"), **{k: np.concatenate(v) for k, v in feats.items()}, y=y)
        os.replace(path.with_suffix(".tmp.npz"), path)
        print(f"{b} {cond} on {name}{seed}: {len(items)} images, {time.time() - t0:.0f}s", flush=True)


def aux(b, name, seed, cond):
    F = np.load(OUT / "aux" / b / f"{name}{seed}_{cond}.npz")
    return np.hstack([F[r].astype(np.float32) for r in layers(b)]), F["y"]


# ---------------------------------------------------------------- the paper's few-shot protocol, in one place
def _mel(Zh, Ze, b, y, split, ks, chosen, ye):
    """Melanoma sensitivity at 80% specificity on ISIC, from the probes at the readout chosen on validation."""
    import warnings
    warnings.filterwarnings("ignore")
    R, Re = candidates(Zh, b), candidates(Ze, b)
    mel = S4.HAM_CLASSES.index("MEL")
    out = {}
    for k in ks:
        X, dims = R[chosen[k]]
        sens = []
        for idx in split["shots"][k][:2 if QUICK else None]:
            s = fit(X[idx], y[idx], [Re[chosen[k]][0]], dims, proba=True)[0][:, mel]
            sens.append(float((s[ye == mel] > np.quantile(s[ye != mel], 0.8)).mean()))
        out[k] = float(np.mean(sens))
    return out


def save(name, C, chosen, **frames):
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"correct_{name}.npz", **C)
    json.dump(chosen, open(OUT / f"chosen_{name}.json", "w"))
    for k, df in frames.items():
        pd.DataFrame(df).to_csv(OUT / f"{k}.csv", index=False)


# ---------------------------------------------------------------- the maps
def mlp(X, Y, seed):
    """A one-hidden-layer MLP stand-in, fit like the linear map, with 20% of the images for early stopping."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    va, tr = idx[:len(X) // 5], idx[len(X) // 5:]
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 1024), torch.nn.GELU(), torch.nn.Linear(1024, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    best, state, bad = np.inf, None, 0
    for _ in range(3 if QUICK else 300):
        net.train()
        for ix in torch.from_numpy(rng.permutation(tr)).split(256):
            loss = ((net(Xt[ix]) - Yt[ix]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(((net(Xt[va]) - Yt[va]) ** 2).mean())
        if v < best - 1e-5:
            best, state, bad = v, {k: x.clone() for k, x in net.state_dict().items()}, 0
        elif (bad := bad + 1) == 10:
            break
    net.load_state_dict(state)
    net.eval()

    def apply(Z):
        with torch.no_grad():
            return np.concatenate([net(torch.from_numpy(Z[i:i + 4096])).numpy() for i in range(0, len(Z), 4096)])
    return apply, best


# ---------------------------------------------------------------- the stand-in and its controls
def standin():
    """The stand-in, its controls, and melanoma sensitivity."""
    from joblib import Parallel, delayed
    sk = Skin()
    T, ye = skin_tests()
    p = sk.X["ham"].shape[1]
    C, chosen = evaluate({"imagenet": sk.X, "imagenet_all": sk.X}, B, T, ye, extra_all=("imagenet_all",))
    fits = []
    todo = {"imagenet": sk.X}
    for s in SEEDS:
        rng = np.random.RandomState(s)
        Y, ys = sk.net(f"plant_disease2_s{s}")
        A, c, alpha = ridge(sk.X["ham"], Y["ham"])
        idx = rng.permutation(len(sk.X["ham"]))
        va, tr = idx[:len(idx) // 5], idx[len(idx) // 5:]
        A2, c2, _ = ridge(sk.X["ham"][tr], Y["ham"][tr], alpha)
        fits.append(dict(seed=s, map="HAM10000 images", alpha=alpha,
                         r2_heldout=r2(Y["ham"][va], sk.X["ham"][va] @ A2 + c2),
                         r2_isic=r2(Y["isic_test"], sk.X["isic_test"] @ A + c)))
        stand = {t: sk.X[t] @ A + c for t in sk.tests}
        sets = {f"plant_disease2_s{s}": Y, f"standin_s{s}": stand}
        for src, img_seed in [("plant", s), ("food", 0)]:
            Xa, Ya = aux(B, src, img_seed, "imagenet")[0], aux(B, src, img_seed, f"plant_disease2_s{s}")[0]
            sx, sy = Std(Xa), Std(Ya)
            Aa, ca, aa = ridge(sx(Xa), sy(Ya))
            Za = {t: sy.inverse(sx(sk.raw[t]) @ Aa + ca) for t in sk.tests}
            fits.append(dict(seed=s, map=f"{src} images", alpha=aa, r2_heldout=np.nan,
                             r2_isic=r2(ys.inverse(Y["isic_test"]), Za["isic_test"])))
            sets[f"standin_{src}fit_s{s}"] = Za
        Fn, _ = sk.net(f"food_s{s}")
        Af, cf, af = ridge(sk.X["ham"], Fn["ham"])
        fits.append(dict(seed=s, map="Food-101 network (HAM10000 images)", alpha=af, r2_heldout=np.nan,
                         r2_isic=r2(Fn["isic_test"], sk.X["isic_test"] @ Af + cf)))
        sets[f"food_s{s}"] = Fn
        sets[f"food_standin_s{s}"] = {t: sk.X[t] @ Af + cf for t in sk.tests}
        rows = rng.choice(len(sk.X["ham"]), int(0.8 * len(sk.X["ham"])), replace=False)
        lam, V = np.linalg.eigh(np.cov(sk.X["ham"][rows], rowvar=False))
        Wz = ((V / np.sqrt(np.maximum(lam, 1e-6 * lam.max()))) @ V.T).astype(np.float32)
        sets[f"zca_s{s}"] = {t: sk.X[t] @ Wz for t in sk.tests}
        mu = spectrum(A, sk.X["ham"])
        Qr = np.linalg.qr(rng.standard_normal((p, p)))[0]
        Tr = ((Qr * np.sqrt(mu)) @ Qr.T).astype(np.float32)
        sets[f"randmap_s{s}"] = {t: sk.X[t] @ Tr for t in sk.tests}
        apply, mse = mlp(sk.X["ham"], Y["ham"], s)
        fits.append(dict(seed=s, map="MLP (HAM10000 images)", alpha=np.nan, r2_heldout=1 - mse / Y["ham"].var(0).mean(),
                         r2_isic=r2(Y["isic_test"], apply(sk.X["isic_test"]))))
        sets[f"mlp_s{s}"] = {t: apply(sk.X[t]) for t in sk.tests}
        c_, ch = evaluate(sets, B, T, ye)
        C.update(c_)
        chosen.update(ch)
        todo.update({f"plant_disease2_s{s}": Y, f"standin_s{s}": stand})
        print(f"seed {s}: {len(sets)} sets scored", flush=True)
    jobs = [(n, sp) for n in todo for sp in SPLITS]
    res = Parallel(n_jobs=S1.cpus())(
        delayed(_mel)(todo[n]["ham"], todo[n]["isic_test"], B, T["ham"]["y"], T["ham"]["splits"][sp], [5, 20, 50],
                      {k: chosen[f"ham|{n}|{sp}|{k}"] for k in [5, 20, 50]}, ye) for n, sp in jobs)
    mel = [dict(network=n, split=sp, k=k, sensitivity=v) for (n, sp), out in zip(jobs, res) for k, v in out.items()]
    save("standin", C, chosen, fits=fits, melanoma=mel)


# ---------------------------------------------------------------- summary
def merged():
    """The stand-in's correctness, {test label: {("s37", name): {(split, k): array}}}, with seedless sets copied to
    every seed so that the helpers of e7_stronger_networks.py (S22) can pair them."""
    Z = np.load(OUT / "correct_standin.npz")
    C = {}
    for key in Z.files:
        t, n, sp, k = key.split("|")
        names = [n] if re.search(r"_s\d+$", n) or n == "imagenet" else [n] + [f"{n}_s{s}" for s in SEEDS]
        for m in names:
            C.setdefault(TESTS[t], {}).setdefault(("s37", m), {})[int(sp), int(k)] = Z[key]
    return C


def labels_saved(base, target):
    """The k at which ImageNet's balanced accuracy reaches the target, interpolated on log k (k = 5, 20, 50)."""
    ks = [5, 20, 50]
    if target <= base[0]:
        return "5 or fewer"
    for (k0, b0), (k1, b1) in zip(zip(ks, base), zip(ks[1:], base[1:])):
        if b0 < target <= b1:
            return f"{np.exp(np.log(k0) + (target - b0) / (b1 - b0) * (np.log(k1) - np.log(k0))):.0f}"
    return "more than 50"


def summarize():
    import e6_other_lesion_targets as G
    T, ye = skin_tests()
    C = merged()
    lab = TESTS
    ks = {lab["isic_test"]: [5, 20, 50], lab["ham"]: [5, 20, 50], **{lab[t]: list(G.KS) for t in INTERNAL}}
    yts = {lab["isic_test"]: {sp: ye for sp in S1.SPLITS}}
    Bx = {lab["isic_test"]: S4._boot_ext(len(ye), ye, SEEDS)}
    for t in ["ham"] + INTERNAL:
        yts[lab[t]] = {sp: T[t]["y"][T[t]["splits"][sp]["te"]] for sp in S1.SPLITS}
        Bx[lab[t]] = S4.BootS(yts[lab[t]], {sp: pd.factorize(T[t]["g"][T[t]["splits"][sp]["te"]])[0]
                                            for sp in S1.SPLITS}, T[t]["nc"], SEEDS)
    nc = {k: int(max(v.max() for v in y.values())) + 1 for k, y in yts.items()}
    ISIC = lab["isic_test"]

    def level(a, test=ISIC, k=None):
        return S22._ba(C[test], ("s37", a), yts[test], nc[test], k or ks[test])

    gains = pd.DataFrame([dict(comparison=f"{a} - {b}", test=test,
                               **S22._cmp(Bx[test], C[test], ("s37", a), ("s37", b), ks[test]))
                          for a, b in CORE for test in C])
    gains["txt"] = [f"{r.gain:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]" for r in gains.itertuples()]
    get = lambda a, b="imagenet": gains[(gains.comparison == f"{a} - {b}") & (gains.test == ISIC)].iloc[0]
    g_pl = get("plant_disease2")
    share = lambda a: get(a).gain / g_pl.gain
    fade = lambda a: ((level(a, k=[5]) - level("imagenet", k=[5])) - (level(a, k=[50]) - level("imagenet", k=[50])))
    fits = pd.read_csv(OUT / "fits.csv")
    rules = pd.DataFrame([
        dict(rule="P1a", test="stand-in gain above zero and at least 2/3 of the plant gain",
             value=f"{get('standin').txt}; share {share('standin'):.2f}",
             holds=get("standin").lo > 0 and share("standin") >= 2 / 3),
        dict(rule="P1b", test="stand-in fades like the plant network (k=5 minus k=50 within 1 point)",
             value=f"{fade('standin'):+.2f} vs {fade('plant_disease2'):+.2f}",
             holds=abs(fade("standin") - fade("plant_disease2")) <= 1),
        dict(rule="P1c", test="stand-in beats ImageNet with an all-layer readout",
             value=get("standin", "imagenet_all").txt, holds=get("standin", "imagenet_all").lo > 0),
        dict(rule="P1d", test="whitening and spectrum-matched random map under 1/3; Food stand-in null",
             value=f"ZCA {share('zca'):.2f}, random {share('randmap'):.2f}, Food stand-in {get('food_standin').txt}",
             holds=share("zca") < 1 / 3 and share("randmap") < 1 / 3 and get("food_standin").lo <= 0),
        dict(rule="P1e", test="stand-ins fit on plant images and on Food-101 photos keep at least half",
             value=f"plant fit {share('standin_plantfit'):.2f}, Food fit {share('standin_foodfit'):.2f}",
             holds=share("standin_plantfit") >= 0.5 and share("standin_foodfit") >= 0.5),
        dict(rule="P1f", test="the MLP stand-in adds under 1/3 of the plant gain",
             value=f"{(get('mlp').gain - get('standin').gain) / g_pl.gain:.2f} of the plant gain",
             holds=(get("mlp").gain - get("standin").gain) < g_pl.gain / 3)])
    mel = pd.read_csv(OUT / "melanoma.csv")
    mel = mel.assign(family=mel.network.str.replace(r"_s\d+$", "", regex=True)).groupby(
        ["family", "k"]).sensitivity.mean().unstack().round(3)
    saved = labels_saved([level("imagenet", k=[k]) for k in [5, 20, 50]], level("plant_disease2", k=[5]))
    levels = pd.DataFrame([dict(network=a, test=test, ba=level(a, test)) for test in C
                           for a in ["imagenet", "imagenet_all", "plant_disease2", "standin", "food"]])
    for name, df in [("verdict", rules), ("gains", gains), ("levels", levels)]:
        df.to_csv(OUT / f"{name}.csv", index=False)
    tests = list(TESTS.values())
    L = ["Leaves to Lesions -- reweighting or new information? The linear stand-in, with its tests fixed before any "
         "run.", "",
         "Map fit (R^2; mean over seeds):\n", fits.groupby("map")[["alpha", "r2_heldout", "r2_isic"]].mean()
         .round(3).to_string(), "",
         "Paired differences in balanced accuracy, points (95% CI):\n",
         gains.pivot(index="comparison", columns="test", values="txt")
         .reindex([f"{a} - {b}" for a, b in CORE])[tests].to_string(), "",
         "Balanced accuracy (%):\n", levels.pivot(index="network", columns="test", values="ba").round(1)[tests]
         .to_string(), "",
         "Melanoma sensitivity at 80% specificity on ISIC:\n", mel.to_string(), "",
         f"Labels saved on ISIC: ImageNet only needs about {saved} labels per class to match plants at k = 5.", "",
         "Pre-registered predictions (part 1):\n", rules.set_index("rule").to_string(), "",
         "Reading: the plant gain is mostly a reweighting of what ImageNet's features already hold: a linear map, fit "
         f"without labels, keeps {100 * share('standin'):.0f}% of the healthy-vs-diseased network's ISIC gain, and "
         f"generic reweightings do not give it ({int(rules.holds.sum())} of {len(rules)} predictions held)."]
    (OUT / "summary.txt").write_text("\n".join(L))
    print("\n".join(L))


# ---------------------------------------------------------------- shared code (common/), with this module's settings
# These call common/ with this module's settings, read at call time (scripts reassign some of them).
def _unit(Z, b, y, split, ks, nc, Ze=None, extra_all=False):
    return _LM.fit_split(Z, b, y, split, ks, nc, Ze, extra_all, draws=2 if QUICK else None)


def skin_tests():
    """Labels and splits: HAM10000 (k = 5/20/50; its shots are also tested on ISIC) and the internal skin tests."""
    T, ye = _LM.skin_tests(INTERNAL)
    S1.KS = [5, 20, 50]                               # leaves the screen's k values at their default
    return T, ye


def evaluate(sets, b, T, ye, extra_all=()):
    """Few-shot correctness of named all-layer feature sets ({name: {test: array}}), flat-keyed test|name|split|k,
    and the readout chosen for each."""
    return _LM.evaluate(sets, b, T, ye, extra_all, SPLITS, INTERNAL, 2 if QUICK else None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["extract", "standin", "summarize"])
    ap.add_argument("--task", type=int)
    a = ap.parse_args()
    if a.step == "extract":
        extract(a.task)
    else:
        {"standin": standin, "summarize": summarize}[a.step]()
