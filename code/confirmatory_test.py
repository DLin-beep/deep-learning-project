"""Do the headline claims replicate on fresh skin data (BCN20000), and do our PanDerm features match PanDerm's paper?

The five claims (C1-C5, see RULES) and their decision rules were fixed before any run. Nothing is trained. The existing
networks are scored with frozen few-shot evaluation, and the PanDerm check uses full-data linear probes on HAM10000.

Steps, each run as its own SLURM job:
  prepare            BCN20000 from the ISIC Archive (collection 249): 8 classes, at most 1,500 images per class
                     drawn by whole lesion, 256 px thumbnails, cached at 224 px (CPU)
  dedup              drop BCN20000 images with a near-duplicate in HAM10000 or the ISIC 2018 test set (GPU)
  features --task i  one network on BCN20000, every condition (GPU; i in 0-2)
  evaluate           few-shot correctness on BCN20000, k = 5 / 20 (CPU)
  probe              PanDerm full-data linear probes on HAM10000 (CPU)
  summarize          -> pilot/out_s31/summary.txt (CPU)
"""
import argparse
import collections
import json
import os
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

for _d in sorted(Path(__file__).resolve().parents[1].glob("[1-9]_*")):   # the paper's nine experiments
    sys.path.insert(0, str(_d))
import e1_pilot as P        # noqa: E402
import source_screen as S1  # noqa: E402
import e2_credibility as S4      # noqa: E402
import e7_stronger_networks as S22    # noqa: E402
import e6_other_lesion_targets as G   # noqa: E402

OUT = P.ROOT / "pilot" / "out_s31"
T = "bcn20000"
API = "https://api.isic-archive.com/api/v2/images/search/?collections=249&limit=100"
CLASSES3 = {"Nevus": "NV", "Melanoma, NOS": "MEL", "Basal cell carcinoma": "BCC", "Seborrheic keratosis": "BKL",
            "Solar lentigo": "BKL", "Solar or actinic keratosis": "AK", "Squamous cell carcinoma, NOS": "SCC",
            "Dermatofibroma": "DF"}
CLASSES2 = {"Benign soft tissue proliferations - Vascular": "VASC"}    # these images have no diagnosis_3
CAP, DUP = 1500, 0.95
SEEDS = S22.SEEDS
NETS = {"vits_dino": ["imagenet", "plant_disease2", "food"],
        "vitb_dinov2": ["imagenet", "plant", "plant_species", "food"],
        "vitl_panderm": ["imagenet", "plant"]}
NAMES = {("vits_dino", "imagenet"): "DINO ViT-S", ("vits_dino", "plant_disease2"): "DINO ViT-S + healthy vs diseased",
         ("vits_dino", "food"): "DINO ViT-S + Food-101", ("vitb_dinov2", "imagenet"): "DINOv2 ViT-B",
         ("vitb_dinov2", "plant"): "DINOv2 ViT-B + PlantVillage 38 classes",
         ("vitb_dinov2", "plant_species"): "DINOv2 ViT-B + species only", ("vitb_dinov2", "food"): "DINOv2 ViT-B + Food-101",
         ("vitl_panderm", "imagenet"): "PanDerm", ("vitl_panderm", "plant"): "PanDerm + PlantVillage"}
COMPS = [(("vits_dino", "plant_disease2"), ("vits_dino", "imagenet")),      # C1
         (("vits_dino", "plant_disease2"), ("vits_dino", "food")),          # C2
         (("vitb_dinov2", "plant"), ("vitb_dinov2", "plant_species")),      # C3
         (("vitb_dinov2", "plant"), ("vitb_dinov2", "food")),               # C4
         (("vitl_panderm", "plant"), ("vitl_panderm", "imagenet")),         # C5
         (("vitb_dinov2", "plant"), ("vitb_dinov2", "imagenet")),
         (("vits_dino", "food"), ("vits_dino", "imagenet")),
         (("vitb_dinov2", "food"), ("vitb_dinov2", "imagenet")),
         (("vitl_panderm", "imagenet"), ("vitb_dinov2", "plant"))]
RULES = ["DINO ViT-S, healthy vs diseased beats ImageNet only", "DINO ViT-S, healthy vs diseased beats Food-101",
         "DINOv2 ViT-B, 38 classes beat species-only labels", "DINOv2 ViT-B, 38 classes beat Food-101",
         "PanDerm after PlantVillage beats PanDerm as it comes"]
PANDERM_LP = {"accuracy": 0.831, "weighted F1": 0.825}      # PanDerm's paper, linear probe on HAM10000


def prepare():
    d = P.DATA / T / "thumbs"
    d.mkdir(parents=True, exist_ok=True)
    rows, url = [], API
    while url:
        page = json.load(urllib.request.urlopen(url, timeout=60))
        for x in page["results"]:
            c = x["metadata"].get("clinical", {})
            dx = CLASSES3.get(c.get("diagnosis_3")) if c.get("diagnosis_3") else CLASSES2.get(c.get("diagnosis_2"))
            if dx and c.get("lesion_id") and x["metadata"].get("acquisition", {}).get("image_type") == "dermoscopic":
                rows.append((x["isic_id"], x["files"]["thumbnail_256"]["url"], dx, c["lesion_id"]))
        url = page.get("next")
    rng, by, keep = random.Random(0), collections.defaultdict(lambda: collections.defaultdict(list)), []
    for r in rows:
        by[r[2]][r[3]].append(r)
    for dx in sorted(by):                              # whole lesions, in a seeded order, until the class reaches CAP
        lesions = sorted(by[dx])
        rng.shuffle(lesions)
        n = 0
        for les in lesions:
            if n >= CAP:
                break
            keep += by[dx][les]
            n += len(by[dx][les])

    def fetch(r):
        path = d / f"{r[0]}.jpg"
        if not path.exists():
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(r[1], tmp)
            tmp.replace(path)
        return str(path)

    with ThreadPoolExecutor(8) as pool:
        paths = list(pool.map(fetch, keep))
    classes = sorted({r[2] for r in keep})
    tab = pd.DataFrame(dict(path=paths, y=[classes.index(r[2]) for r in keep], group=[r[3] for r in keep],
                            isic_id=[r[0] for r in keep]))
    print(f"BCN20000: {len(rows)} labelled dermoscopic images in the 8 classes, {len(keep)} kept", flush=True)
    G.cache(T, tab, classes)


def _unit_cls(path):
    X = S1.readout(np.load(path), "cls", "vits_dino")[0]
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def dedup():
    """Near-duplicates of HAM10000 or ISIC 2018 test images (DINO ViT-S ImageNet CLS, cosine >= 0.95) leave the set."""
    S4.extract_conditions("vits_dino", T, ["imagenet"])
    own = S4.target_dir("vits_dino", T) / "feats_imagenet.npz"
    B = _unit_cls(own)
    R = np.concatenate([_unit_cls(S1.OUT / "vits_dino" / "ham" / "feats_imagenet.npz"),
                        _unit_cls(S4.target_dir("vits_dino", "isic_test") / "feats_imagenet.npz")])
    best = np.concatenate([(B[i:i + 1024] @ R.T).max(1) for i in range(0, len(B), 1024)])
    drop = best >= DUP
    OUT.mkdir(parents=True, exist_ok=True)
    rep = dict(images=int(len(B)), dropped=int(drop.sum()), highest=float(best.max()),
               quantiles={q: float(np.quantile(best, q)) for q in (0.5, 0.9, 0.99)})
    json.dump(rep, open(OUT / "dedup.json", "w"), indent=1)
    print("duplicate check:", rep, flush=True)
    if drop.any():                                     # rewrite the cache without them; features are then recomputed
        d = S4.OUT / "cache"
        tab = pd.read_csv(d / f"{T}_index.csv")
        tab[~drop].to_csv(d / f"{T}_index.csv", index=False)
        old = np.load(d / f"{T}_u8.npy", mmap_mode="r")
        np.save(d / f"{T}_u8.tmp.npy", np.asarray(old[np.flatnonzero(~drop)]))
        del old
        os.replace(d / f"{T}_u8.tmp.npy", d / f"{T}_u8.npy")
        own.unlink()


def features(task):
    b = list(NETS)[task]
    S4.extract_conditions(b, T, S22.conds(b))


def probe():
    """Full-data linear probes of PanDerm as it comes on HAM10000."""
    import warnings
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    warnings.filterwarnings("ignore")
    F = np.load(S1.OUT / "vitl_panderm" / "ham" / "feats_imagenet.npz")
    tab, _ = S1.load_target("ham")
    y, g = tab["y"].values, tab["group"].values
    splits = S1.make_splits(y, g, 7)
    perm = np.random.RandomState(0).permutation(len(y))
    image_split = {"tr": perm[:int(0.8 * len(y))], "te": perm[int(0.8 * len(y)):]}

    def fit(X, tr, te, C, weight):
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd < 1e-8] = 1.0
        clf = LogisticRegression(C=C, max_iter=5000, class_weight=weight).fit((X[tr] - mu) / sd, y[tr])
        p = clf.predict((X[te] - mu) / sd)
        return dict(accuracy=accuracy_score(y[te], p), weighted_f1=f1_score(y[te], p, average="weighted"),
                    balanced_accuracy=balanced_accuracy_score(y[te], p))

    rows = []
    for r in ("cls", "cat"):
        X = S1.readout(F, r, "vitl_panderm")[0]
        for sp, s in splits.items():
            rows.append(dict(split="lesion-grouped", readout=r, C=0.1, part=sp,
                             **fit(X, np.concatenate([s["itr"], s["val"]]), s["te"], 0.1, "balanced")))
        rows.append(dict(split="image-level 80/20", readout=r, C=1.0, part=0,
                         **fit(X, image_split["tr"], image_split["te"], 1.0, None)))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "panderm_probe.csv", index=False)
    print(pd.DataFrame(rows).groupby(["split", "readout", "C"])[["accuracy", "weighted_f1", "balanced_accuracy"]]
          .mean().round(3).to_string(), flush=True)


def summarize():
    Z = np.load(OUT / f"correct_{T}.npz")
    yt = {sp: Z[f"y|{sp}"] for sp in S1.SPLITS}
    C = {(b, c): {(sp, k): Z[f"{b}|{c}|{sp}|{k}"] for sp in S1.SPLITS for k in G.KS}
         for b in NETS for c in S22.conds(b)}
    nc = int(max(v.max() for v in yt.values())) + 1
    Bx = S4.BootS(yt, {sp: Z[f"g|{sp}"] for sp in S1.SPLITS}, nc, SEEDS)
    levels = pd.DataFrame([dict(network=NAMES[x], ba=S22._ba(C, x, yt, nc, G.KS)) for x in NAMES])
    gains = pd.DataFrame([dict(comparison=f"{NAMES[a]} - {NAMES[b]}", **S22._cmp(Bx, C, a, b, G.KS))
                          for a, b in COMPS])
    rules = pd.DataFrame([dict(rule=f"C{i + 1}", test=RULES[i], holds=bool(gains.lo.iloc[i] > 0)) for i in range(5)])
    for name, df in [("levels", levels), ("gains", gains), ("verdict", rules)]:
        df.to_csv(OUT / f"{name}.csv", index=False)
    dd = json.load(open(OUT / "dedup.json"))
    classes = json.load(open(G.OUT / f"{T}_classes.json"))
    tab = pd.read_csv(S4.OUT / "cache" / f"{T}_index.csv")
    pr = pd.read_csv(OUT / "panderm_probe.csv").groupby(["split", "readout", "C"])[
        ["accuracy", "weighted_f1", "balanced_accuracy"]].mean().round(3)
    n_held = int(rules.holds.sum())
    gains["txt"] = [f"{r.gain:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}] {r.cells}" for r in gains.itertuples()]
    L = ["Leaves to Lesions -- a confirmatory test on BCN20000, never used before (frozen few-shot, k = 5 / 20; tests "
         "fixed before any run).", "",
         f"BCN20000: {len(tab)} images, {tab.group.nunique()} lesions; per class "
         f"{dict(zip(classes, np.bincount(tab.y, minlength=len(classes)).tolist()))}.",
         f"Duplicate check: {dd['dropped']} of {dd['images']} images dropped (cosine >= {DUP} to HAM10000 or ISIC 2018 "
         f"test); highest similarity {dd['highest']:.3f}, median {dd['quantiles']['0.5']:.3f}.", "",
         "Balanced accuracy (%), mean over k:\n", levels.set_index("network").round(1).to_string(), "",
         "Paired differences in points (95% CI, seed x split cells positive):\n",
         gains.set_index("comparison")[["txt"]].to_string(header=False), "",
         "Pre-registered predictions (BCN20000):\n", rules.set_index("rule").to_string(), "",
         f"Reading: {n_held} of 5 headline claims replicate on data never used before.", "",
         "PanDerm as it comes, full-data linear probes on HAM10000 (PanDerm's paper: accuracy "
         f"{PANDERM_LP['accuracy']:.3f}, weighted F1 {PANDERM_LP['weighted F1']:.3f}):\n", pr.to_string()]
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "dedup", "features", "evaluate", "probe", "summarize"])
    ap.add_argument("--task", type=int)
    a = ap.parse_args()
    S22.NETS, S22.OUT, S22.SKIN_TESTS = NETS, OUT, [T]
    if a.step == "features":
        features(a.task)
    else:
        {"prepare": prepare, "dedup": dedup, "evaluate": S22.skin, "probe": probe, "summarize": summarize}[a.step]()
