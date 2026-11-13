"""Does teaching DINO ViT-S/16 to point at the anomaly make its abnormality signal sharper and less tied to plants?

Sharper means higher few-shot accuracy on ISIC, and less tied to plants means gains outside skin. Three tests run in
order. The first asks whether a pair loss on matched images beats ordinary labels. The second asks whether patch-level
supervision from synthetic lesion masks adds to the pairs. The third asks whether hiding which source an image came
from helps a mixture of sources. No skin image is used in training. The tests, predictions and gates were fixed
before any code.

Each network trains with the paper's source recipe under a new objective. Blocks 7-12, the final norm and the new
heads train with SGD (learning rate 0.01, momentum 0.9, no weight decay, clipping at 1) on a cosine schedule over 840
steps of 64 images. Augmentation is a random resized crop and flip, and the seeds are 0-5. The paper's frozen few-shot
protocol then scores each network. A family passes the learnt check when every seed's held-out pair accuracy reaches
0.75.

Families, with their labels in the results:
  pairs_species      P1           species-matched PlantVillage pairs, loss -log sigmoid(s(diseased) - s(healthy))
  pairs_species_bce  P1-ctrl      the same images and batches, binary cross-entropy on each image
  pairs_synth        P2           healthy leaves and COIL-100 views, each with a copy carrying one synthetic lesion
  unpaired_synth     P2-unpaired  the same images with binary cross-entropy, an image and its copy never in one batch
  pairs_synth_patch  SP           P2 plus a patch head on the final-block tokens, trained against the lesion masks
  mix3               MIX          PlantVillage, cropped VisA and synthetic lesions on COIL-100 (900 pairs each) under
                                  the objective the gate carries forward, with a source head that does not push back
  mix3_adv           ADV          MIX with the source head behind gradient reversal (the DANN schedule)
Baselines, reused: imagenet (B0) and plant_disease2_s0-5 (B1).

Steps, each run as its own SLURM job:
  selftest           blending, lesions, samplers, losses, gradient reversal and scoring on synthetic data (CPU)
  make --task s      seed s's synthetic lesions and masks (CPU; s in 0-5). Task 0 also makes the lesions of the probe
                     set and the learnt check, and VisA's defect masks cropped like its images
  check              checks that an augmented synthetic pair differs only inside its lesion, and saves a contact
                     sheet (CPU)
  train --task i     one condition x seed (GPU; 0-29 for the first two tests, 30-41 for the third)
  features --task i  one network: the six tests, the ISIC 2018 Task 1 patch tokens and the probe set (GPU; 0-36 for
                     the baselines and the first two tests, 37-48 for the third)
  evaluate --task i  one network: few-shot on the five targets, plus ISIC from the HAM10000 shots (CPU)
  diagnose --task i  one network: robust-PCA localization, source decodability, linear-map retention (CPU)
  gate               scores the first two tests and picks the third test's objective -> out_s45/gate.json (CPU)
  summarize          -> pilot/out_s45/summary.txt (CPU)

Checkpoints, and the features and predictions on HAM10000, DDR and ISIC, go where the pipeline keeps them (S1.OUT,
S4.OUT) under the new condition names. The other targets' predictions, the heads, the diagnostics, the synthetic
images, the tables and the manifests go to pilot/out_s45.

E15_QUICK=1 runs a smoke test into pilot/out_s45/quick: a few images of every source, two training steps of every
condition, 16 images of each test, the Task 1 tokens with robust PCA on 10 images, one split and two draws. Its
scoring maps every new condition to the plant networks, so the gate and the summary run end to end on real
predictions.
"""
import argparse
import functools
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

_PILOT = Path(__file__).resolve().parents[1]
for _d in [_PILOT.parent] + sorted((_PILOT.parent / "archive" / "paper-v1").glob("[1-9]_*")) + [_PILOT / "10_classifiers"]:
    sys.path.insert(0, str(_d))                  # common/, the paper's nine experiments, then 10_classifiers
QUICK = os.environ.get("E15_QUICK") == "1"
if QUICK:
    os.environ["S37_QUICK"] = "1"              # the linear stand-in's helpers also run one split, two draws
import e1_pilot as P        # noqa: E402
import e1_pilot2 as Q       # noqa: E402
import e1_screen as S1      # noqa: E402
import e2_credibility as S4      # noqa: E402
import e9_linear_stand_in as E9  # noqa: E402  (Std, load, ridge, r2, skin_tests, evaluate)
import e10_classifiers as E10    # noqa: E402  (code_version, isic_units, target, save, Boot40)

OUT = P.ROOT / "pilot" / "out_s45" / ("quick" if QUICK else "")
S41 = P.ROOT / "pilot" / "out_s41"            # the robust-PCA test's Task 1 masks, and its scores for the smoke test
B, BASE = "vits_dino", "plant_disease2"
SEEDS = [0, 1, 2, 3, 4, 5]
SCORE_SEEDS = [0, 1] if QUICK else SEEDS
TEST1 = ["pairs_species", "pairs_species_bce", "pairs_synth", "unpaired_synth"]
TEST2 = ["pairs_synth_patch"]
TEST3 = ["mix3", "mix3_adv"]
FAMILIES = TEST1 + TEST2 + TEST3
CODE = {"imagenet": "B0", BASE: "B1", "pairs_species": "P1", "pairs_species_bce": "P1-ctrl", "pairs_synth": "P2",
        "unpaired_synth": "P2-unpaired", "pairs_synth_patch": "SP", "mix3": "MIX", "mix3_adv": "ADV"}
RUNS = [(f, s) for f in FAMILIES for s in SEEDS]
NETS = ["imagenet"] + [f"{BASE}_s{s}" for s in SEEDS] + [f"{f}_s{s}" for f, s in RUNS]
TESTS = ["ham", "isic_test", "ddr", "hyperkvasir6", "pad_ufes20", "hiba_derm"]
TARGETS = ["ham", "ddr", "hyperkvasir6", "pad_ufes20", "hiba_derm"]         # the five with validation splits
OTHER = ["hyperkvasir6", "pad_ufes20", "hiba_derm"]                         # their predictions go to out_s45
OTHER_KS = [5, 20]                             # the rarest classes allow k = 5 and 20
LABEL = {"ham": "HAM10000", "isic_test": "ISIC 2018 test", "ddr": "DDR", "hyperkvasir6": "HyperKvasir",
         "pad_ufes20": "PAD-UFES-20", "hiba_derm": "HIBA"}
SPLITS = S1.SPLITS[:1] if QUICK else S1.SPLITS
DRAWS = 2 if QUICK else S1.REPEATS
PV = P.DATA / "plantvillage" / "repo" / "raw" / "segmented"
COIL = P.DATA / "coil100" / "coil-100"
VISA = P.DATA / "visa256crop"
VISA_MASKS = OUT / "visa_masks"
SPECIES = ["Apple", "Cherry_(including_sour)", "Corn_(maize)", "Grape", "Peach", "Pepper,_bell", "Potato",
           "Strawberry", "Tomato"]             # the nine PlantVillage species with healthy and diseased leaves
PAIRS_PER_SPECIES, N_LEAF, N_COIL = (4, 16, 16) if QUICK else (300, 1350, 1350)
MIX_PER_SPECIES, MIX_PER_TYPE, MIX_PER_OBJECT = (2, 2, 9) if QUICK else (100, 75, 9)
N_HELD_PER_SPECIES, N_HELD_SYN, N_PROBE = (1, 8, 10) if QUICK else (20, 100, 400)
RESERVE_SEED = 15                              # draws the images no condition trains on
LESION_AREA, VERTICES, RADII = (0.01, 0.20), 12, (0.6, 1.4)
BRIGHT, CHANNEL, FEATHER = (0.6, 1.4), (0.85, 1.15), 3.0   # the colour shift, and its ramp inside the mask (px)
KEEP, CROP_TRIES = 0.5, 10                     # a crop is redrawn until it keeps half the lesion, at most 10 tries
PATCH_POS, PATCH_W, GRID = 0.25, 1.0, 14       # a token is positive when its 16-px patch is at least a quarter lesion
BATCH_PAIRS = 8 if QUICK else P.BATCH // 2
STEPS = 2 if QUICK else P.EPOCHS * 84          # 840: 10 epochs of 84 batches of 64 images, as in the paper's runs
LEARNT, GAP, RETAIN, DECODE_DROP = 0.75, 0.3, 0.80, 15.0
TOL, MAX_ITER, RHO = 1e-7, 1000, 1.5           # robust PCA settings, as in the robust-PCA test
N_LOC_QUICK, N_QUICK = 10, 16
CKPT = OUT / "ckpt" if QUICK else S1.OUT / B / "ckpt"
CONFIG = dict(quick=QUICK, backbone=B, seeds=SEEDS, splits=SPLITS, draws=DRAWS, ks=S1.KS, other_ks=OTHER_KS, c_lr=Q.C,
              pairs_per_species=PAIRS_PER_SPECIES, n_leaf=N_LEAF, n_coil=N_COIL, mix=(MIX_PER_SPECIES, MIX_PER_TYPE,
              MIX_PER_OBJECT), held=(N_HELD_PER_SPECIES, N_HELD_SYN), probe=N_PROBE, reserve_seed=RESERVE_SEED,
              lesion_area=LESION_AREA, vertices=VERTICES, radii=RADII, brightness=BRIGHT, channel=CHANNEL,
              feather=FEATHER, keep=KEEP, crop_tries=CROP_TRIES, patch_pos=PATCH_POS, patch_w=PATCH_W,
              batch_pairs=BATCH_PAIRS, steps=STEPS, lr=P.LR, learnt=LEARNT, gap=GAP, retain=RETAIN,
              decode_drop=DECODE_DROP, rpca=(TOL, MAX_ITER, RHO))


# ---------------------------------------------------------------- files
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(step, task=None, **extra):
    """Save the code version, the hashes of this file and of e1_screen.py, the configuration and the environment with
    every output, in OUT/manifests/<step>_<task>.json."""
    from importlib import metadata
    versions = {}
    for p in ["numpy", "scipy", "scikit-learn", "scikit-image", "pandas", "torch", "torchvision", "timm", "joblib"]:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    m = dict(step=step, task=task, code_version=E10.code_version(), script_sha256=sha256(__file__),
             screen_sha256=sha256(S1.__file__), config=CONFIG, python=platform.python_version(), packages=versions,
             host=platform.node(), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
             slurm_job=os.environ.get("SLURM_JOB_ID"), slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"),
             finished=time.strftime("%Y-%m-%d %H:%M:%S"), **extra)
    d = OUT / "manifests"
    d.mkdir(parents=True, exist_ok=True)
    name = step if task is None else f"{step}_{task}"
    json.dump(m, open(d / f"{name}.json", "w"), indent=1, default=str)


def family(c):
    return c if c == "imagenet" else c.rsplit("_s", 1)[0]


def cond(fam, s=None):
    """A family's network of seed s. The smoke test scores every new family with the plant networks."""
    if fam == "imagenet":
        return "imagenet"
    return f"{BASE if QUICK else fam}_s{s}"


def heads_path(c):
    return OUT / "heads" / f"{c}.pt"


def mask_path(p):
    p = Path(p)
    return str(p.with_name(p.stem + "_mask.png"))


def synth_index(name):
    return pd.read_csv(OUT / "synth" / name / "index.csv")


def visa_crop(p):
    """A visa256 path's copy cropped to the object (S1._visa_crop)."""
    return str(VISA / Path(p).relative_to(P.DATA / "visa256"))


def visa_mask(p):
    """A cropped defective VisA photo's mask, cropped the same way (made by visa_masks)."""
    return str(VISA_MASKS / Path(p).relative_to(VISA).with_suffix(".png"))


# ---------------------------------------------------------------- source images
def quota(sizes, n, rng):
    """Counts per key that sum to n, as even as sizes ({key: images available}) allows. A key short of its share
    gives all it has, and the rest is split again. Remainders go to keys drawn with rng."""
    counts, left = dict.fromkeys(sizes, 0), n
    open_ = [k for k in sizes if sizes[k] > 0]
    while left > 0 and open_:
        q, extra = divmod(left, len(open_))
        plus = set(rng.sample(open_, extra))
        for k in open_:
            counts[k] += min(q + (k in plus), sizes[k] - counts[k])
        left = n - sum(counts.values())
        open_ = [k for k in open_ if counts[k] < sizes[k]]
    assert left == 0, f"only {n - left} of {n} images available"
    return counts


@functools.lru_cache(None)
def plant_files():
    """PlantVillage's segmented leaves (B1's images), {class: sorted paths}."""
    return {c: sorted(str(p) for p in (PV / c).iterdir() if p.suffix.lower() in S1.IMG_EXTS)
            for c in S1._class_dirs(PV)}


def species(c):
    return c.split("___")[0]


def is_healthy(c):
    return "healthy" in c


@functools.lru_cache(None)
def coil_files():
    """COIL-100's views, {object: sorted paths} (72 each)."""
    per = {}
    for p in sorted(COIL.glob("obj*__*.png")):
        per.setdefault(int(p.name.split("__")[0][3:]), []).append(str(p))
    return per


@functools.lru_cache(None)
def reserved():
    """The images no condition trains on, drawn with RESERVE_SEED and the same for every seed. They are the learnt
    check's PlantVillage pairs (per paired species), the probe set's PlantVillage images (half healthy, half diseased)
    and COIL-100 views, and the leaves and views that get the learnt check's synthetic lesions."""
    r, Fl, C = random.Random(RESERVE_SEED), plant_files(), coil_files()
    used = set()

    def take(files, n):
        pick = r.sample([f for f in files if f not in used], n)
        used.update(pick)
        return pick

    def spread(groups, files, n):
        cnt = quota({g: len([f for f in files[g] if f not in used]) for g in groups}, n, r)
        return [f for g in groups for f in take(files[g], cnt[g])]

    healthy = [c for c in Fl if is_healthy(c)]
    diseased = [c for c in Fl if not is_healthy(c)]
    held_pv = []
    for sp in SPECIES:
        h = take(Fl[f"{sp}___healthy"], N_HELD_PER_SPECIES)
        d = spread([c for c in diseased if species(c) == sp], Fl, N_HELD_PER_SPECIES)
        r.shuffle(d)
        held_pv += list(zip(h, d))
    probe_pv = ([(f, 0) for f in spread(healthy, Fl, N_PROBE // 2)] +
                [(f, 1) for f in spread(diseased, Fl, N_PROBE - N_PROBE // 2)])
    held_leaf = spread(healthy, Fl, N_HELD_SYN)
    objs = sorted(C)
    probe_coil = spread(objs, C, N_PROBE)
    held_coil = spread(objs, C, N_HELD_SYN)
    return dict(held_pv=held_pv, probe_pv=probe_pv, held_leaf=held_leaf, probe_coil=probe_coil, held_coil=held_coil,
                used=frozenset(used))


def unreserved(files):
    used = reserved()["used"]
    return [f for f in files if f not in used]


def species_pairs(rng, per_species):
    """per_species (healthy, diseased) pairs for each of the nine species with both, from unreserved leaves. Diseased
    leaves are spread as evenly over the species' diseases as possible. Healthy leaves are drawn without replacement
    while they last (at 300 pairs, each potato leaf is used about three times)."""
    Fl, out = plant_files(), []
    for sp in SPECIES:
        H = unreserved(Fl[f"{sp}___healthy"])
        hs = []
        while len(hs) < per_species:
            hs += rng.sample(H, min(len(H), per_species - len(hs)))
        pools = {c: unreserved(Fl[c]) for c in Fl if species(c) == sp and not is_healthy(c)}
        cnt = quota({c: len(v) for c, v in pools.items()}, per_species, rng)
        ds = [f for c in pools for f in rng.sample(pools[c], cnt[c])]
        rng.shuffle(ds)
        out += [dict(normal=h, abnormal=d, mask=None, source=0) for h, d in zip(hs, ds)]
    rng.shuffle(out)
    return out


def synth_bases(seed):
    """A seed's clean images as [(path, kind)]. They are N_LEAF healthy leaves, spread as evenly over the 12 healthy
    classes as the unreserved leaves allow, and N_COIL COIL-100 views, as evenly over the 100 objects as possible."""
    rng, Fl, C = random.Random(seed), plant_files(), coil_files()
    leaf = {c: unreserved(Fl[c]) for c in Fl if is_healthy(c)}
    cnt = quota({c: len(v) for c, v in leaf.items()}, N_LEAF, rng)
    out = [(f, "leaf") for c in leaf for f in rng.sample(leaf[c], cnt[c])]
    view = {o: unreserved(C[o]) for o in sorted(C)}
    cnt = quota({o: len(v) for o, v in view.items()}, N_COIL, rng)
    return out + [(f, "coil") for o in view for f in rng.sample(view[o], cnt[o])]


def visa_table():
    """VisA (visa256) with object types, defect flags and the fixed held-out set of the abnormality-sources test."""
    return S1._abnormality_table("visa_defect")


def visa_pairs(rng, per_type, held=False):
    """Pairs of a normal and a defective VisA photo of the same object type, cropped to the object, with the defect
    masks. By default there are per_type pairs per type, from outside the fixed held-out set. With held=True, every
    held-out defective photo is paired with a held-out normal photo of its type."""
    df = visa_table()
    df = df[df.heldout] if held else df[~df.heldout]
    out = []
    for o in sorted(df.group.unique()):
        d = sorted(df[(df.group == o) & df.abnormal].path)
        n = sorted(df[(df.group == o) & ~df.abnormal].path)
        d = d[:1] if held and QUICK else (d if held else rng.sample(d, per_type))
        out += [dict(normal=visa_crop(a), abnormal=visa_crop(b), mask=visa_mask(visa_crop(b)), source=1)
                for a, b in zip(rng.sample(n, len(d)), d)]
    return out


def mix3_data(seed):
    """The third test's 2,700 pairs, 900 from each source. PlantVillage pairs are matched by species and VisA pairs by
    object type. The synthetic pairs reuse the seed's pairs_synth lesions on COIL-100, the first nine views of each
    object."""
    S = synth_index(f"s{seed}")
    coil = S[S.kind == "coil"].assign(obj=lambda d: [Path(b).name.split("__")[0] for b in d.base])
    coil = coil.groupby("obj", sort=False).head(MIX_PER_OBJECT)
    synth = [dict(normal=r.base, abnormal=r.lesioned, mask=r.mask, source=2) for r in coil.itertuples()]
    return (species_pairs(random.Random(1000 + seed), MIX_PER_SPECIES) +
            visa_pairs(random.Random(2000 + seed), MIX_PER_TYPE) + synth)


def train_data(fam, seed):
    if fam in ("pairs_species", "pairs_species_bce"):
        return species_pairs(random.Random(seed), PAIRS_PER_SPECIES)
    if fam in TEST3:
        return mix3_data(seed)
    return [dict(normal=r.base, abnormal=r.lesioned, mask=r.mask, source=2) for r in synth_index(f"s{seed}").itertuples()]


@functools.lru_cache(None)
def held_sets():
    """The learnt check's held-out pairs: PlantVillage (reserved), synthetic lesions on reserved leaves and views,
    and VisA's held-out set."""
    H = synth_index("held")
    synth = [dict(normal=r.base, abnormal=r.lesioned, mask=r.mask, source=2) for r in H.itertuples()]
    return dict(plants=[dict(normal=h, abnormal=d, mask=None, source=0) for h, d in reserved()["held_pv"]],
                synthetic=synth, coil=[p for p, k in zip(synth, H.kind) if k == "coil"],
                visa=visa_pairs(random.Random(RESERVE_SEED), 0, held=True))


def held_pairs(fam):
    h = held_sets()
    if fam.startswith("pairs_species"):
        return {"plants": h["plants"]}
    if fam in TEST3:
        return {"plants": h["plants"], "visa": h["visa"], "synthetic": h["coil"]}
    return {"synthetic": h["synthetic"]}


@functools.lru_cache(None)
def probe_items():
    """The probe set as (path, source, abnormal), N_PROBE images per source, half normal and half abnormal. The
    sources are reserved PlantVillage images, VisA photos from its fixed held-out set (cropped to the object), and
    reserved COIL-100 views, the second half with a synthetic lesion."""
    r = random.Random(RESERVE_SEED)
    pv = [(f, 0, a) for f, a in reserved()["probe_pv"]]
    df = visa_table()
    df = df[df.heldout]
    visa = []
    for ab in (False, True):
        g = {o: sorted(df[(df.group == o) & (df.abnormal == ab)].path) for o in sorted(df.group.unique())}
        cnt = quota({o: len(v) for o, v in g.items()}, N_PROBE // 2, r)
        visa += [(visa_crop(f), 1, int(ab)) for o in g for f in r.sample(g[o], cnt[o])]
    coil = reserved()["probe_coil"]
    les = synth_index("probe")
    assert list(les.base) == coil[len(coil) // 2:]
    return tuple(pv + visa + [(f, 2, 0) for f in coil[:len(coil) // 2]] + [(f, 2, 1) for f in les.lesioned])


# ---------------------------------------------------------------- synthetic lesions
def object_mask(a):
    import e5_lesion_like_sources as S18           # the blotch test's mask: border colour, opened, hole-filled
    return S18.object_mask(a)


def poisson(dst, src, mask):
    """Normal-clone Poisson blending (Perez et al. 2003), the equation OpenCV's seamlessClone solves.

    Inside the mask the result has src's gradients and meets dst along the mask's edge. Outside it is dst, unchanged.
    It is solved exactly (sparse LU) on the 4-neighbour grid. The mask must not touch the image border. dst and src
    are h x w x 3 in [0, 1]."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import splu
    ys, xs = np.nonzero(mask)
    n = len(ys)
    index = np.full(mask.shape, -1)
    index[ys, xs] = np.arange(n)
    rows, cols, vals = [np.arange(n)], [np.arange(n)], [np.full(n, 4.0)]
    rhs = np.zeros((n, 3))
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny, nx = ys + dy, xs + dx
        rhs += src[ys, xs] - src[ny, nx]                 # the guidance field: src's gradients
        inside = mask[ny, nx]
        rows.append(np.flatnonzero(inside))
        cols.append(index[ny[inside], nx[inside]])
        vals.append(-np.ones(int(inside.sum())))
        rhs[~inside] += dst[ny[~inside], nx[~inside]]    # the edge condition: dst's values
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsc()
    out = dst.copy()
    out[ys, xs] = np.clip(splu(A).solve(rhs), 0, 1)
    return out


def lesion(a, other, rng):
    """One synthetic lesion on image a, filled from image `other` (h x w x 3 floats in [0, 1], other at a's size).

    The lesion is an irregular blob (a smoothed polygon of VERTICES vertices at random radii), wholly inside the
    object, with an area drawn from LESION_AREA as a share of the object's. A patch of `other`, centred on one of its
    object pixels, is Poisson-blended into it. A colour shift (brightness and per-channel gains) then applies inside
    the mask, ramping in over FEATHER px. Returns (image, mask, record), and the image equals a outside the mask."""
    from scipy import ndimage
    from skimage.draw import polygon
    h, w, _ = a.shape
    obj = object_mask(a)
    if obj.sum() < 0.02 * h * w:                           # no object found: the central half of the image
        obj = np.zeros((h, w), bool)
        obj[h // 4:3 * h // 4, w // 4:3 * w // 4] = True
    room = ndimage.binary_erosion(obj, iterations=2)
    room[:2], room[-2:], room[:, :2], room[:, -2:] = False, False, False, False    # the blend reads a ring around it
    depth = ndimage.distance_transform_edt(room)
    for _ in range(10):
        target = rng.uniform(*LESION_AREA) * obj.sum()
        ang = rng.uniform(0, 2 * np.pi) + 2 * np.pi * np.arange(VERTICES) / VERTICES
        rel = rng.uniform(*RADII, VERTICES)

        def blob(r0, cy, cx):
            m = np.zeros((h, w), bool)
            rr, cc = polygon(cy + r0 * rel * np.sin(ang), cx + r0 * rel * np.cos(ang), (h, w))
            m[rr, cc] = True
            return ndimage.gaussian_filter(m.astype(float), max(0.15 * r0, 0.5)) > 0.5

        r0 = np.sqrt(target / np.pi)
        r0 *= np.sqrt(target / max(blob(r0, h / 2, w / 2).sum(), 1))           # one correction to the target area
        fits = np.argwhere(depth >= r0 * rel.max())
        cy, cx = fits[rng.randint(len(fits))] if len(fits) else np.unravel_index(depth.argmax(), depth.shape)
        mask = blob(r0, cy, cx) & room
        if mask.sum() >= 16:
            break
    else:                                                  # a thin object: a small disc at its deepest point
        cy, cx = np.unravel_index(depth.argmax(), depth.shape)
        yy, xx = np.mgrid[:h, :w]
        mask = (np.hypot(yy - cy, xx - cx) <= max(2.5, min(depth.max(), np.sqrt(target / np.pi)))) & room
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min() - 1, ys.max() + 2, xs.min() - 1, xs.max() + 2     # the box and the ring the blend reads
    hh, ww = y1 - y0, x1 - x0
    where = np.argwhere(object_mask(other))
    py, px = where[rng.randint(len(where))] if len(where) else (h // 2, w // 2)
    sy, sx = int(np.clip(py - hh // 2, 0, h - hh)), int(np.clip(px - ww // 2, 0, w - ww))
    src = a.copy()
    src[y0:y1, x0:x1] = other[sy:sy + hh, sx:sx + ww]
    out = poisson(a, src, mask)
    bright, gain = rng.uniform(*BRIGHT), rng.uniform(*CHANNEL, 3)
    alpha = np.clip(ndimage.distance_transform_edt(mask) / FEATHER, 0, 1)[..., None]   # 0 outside the mask
    out = out * (1 - alpha) + np.clip(out * bright * gain, 0, 1) * alpha
    return out, mask, dict(object_px=int(obj.sum()), target_px=round(float(target)), lesion_px=int(mask.sum()),
                           brightness=float(bright), gain_r=float(gain[0]), gain_g=float(gain[1]),
                           gain_b=float(gain[2]))


def _lesion_job(job):
    """Worker: write one lesioned copy and its mask as PNG. PNG is lossless, so the copy equals its clean image outside
    the lesion."""
    from PIL import Image
    base, other, dst, seed = job
    rng = np.random.RandomState(seed)
    a = np.asarray(Image.open(base).convert("RGB"), dtype=np.float64) / 255
    b = Image.open(other).convert("RGB")
    if b.size != (a.shape[1], a.shape[0]):
        b = b.resize((a.shape[1], a.shape[0]), Image.BILINEAR)
    out, mask, rec = lesion(a, np.asarray(b, dtype=np.float64) / 255, rng)
    Image.fromarray((out * 255).round().astype(np.uint8)).save(dst)
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path(dst))
    return rec


def synth_set(name, bases, seed, fills=None):
    """Lesioned copies of `bases` [(path, kind)] -> out_s45/synth/<name>/ (NNNN.png, NNNN_mask.png, index.csv). Each
    lesion is filled from another image drawn from `fills` (the bases by default)."""
    d = OUT / "synth" / name
    if (d / "index.csv").exists():
        print("exists:", d); return
    d.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    fills = list(fills or [f for f, _ in bases])
    others = []
    for f, _ in bases:
        o = f
        while o == f:
            o = fills[rng.randrange(len(fills))]
        others.append(o)
    jobs = [(f, o, str(d / f"{i:04d}.png"), seed * 100000 + i) for i, ((f, _), o) in enumerate(zip(bases, others))]
    with Pool(S1.cpus()) as pool:
        recs = pool.map(_lesion_job, jobs, chunksize=8)
    df = pd.DataFrame([dict(i=i, base=f, kind=k, other=o, lesioned=j[2], mask=mask_path(j[2]), **r)
                       for i, ((f, k), o, j, r) in enumerate(zip(bases, others, jobs, recs))])
    df.to_csv(d / "index.tmp.csv", index=False)
    os.replace(d / "index.tmp.csv", d / "index.csv")
    share = df.lesion_px / df.object_px
    print(f"{name}: {len(df)} lesions; lesion share of the object median {share.median():.3f} "
          f"[{share.quantile(0.05):.3f}, {share.quantile(0.95):.3f}]; below 16 px: {(df.lesion_px < 16).sum()}",
          flush=True)
    sheet(df.head(12), OUT / "sheets" / f"lesions_{name}.jpg")


def sheet(rows, dst):
    """A contact sheet: clean image, lesioned copy and mask, one pair a row."""
    from PIL import Image
    dst.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows.itertuples())
    out = Image.new("RGB", (3 * 160, 160 * len(rows)))
    for i, r in enumerate(rows):
        for j, p in enumerate((r.base, r.lesioned, r.mask)):
            out.paste(Image.open(p).convert("RGB").resize((160, 160)), (160 * j, 160 * i))
    out.save(dst, quality=90)


def _visa_mask(job):
    """Worker: crop one defective VisA photo's mask with the box e5_defect_crop._crop_visa computes from the photo,
    and resize it to the cropped photo's size (nearest). Returns the defect's share of the crop, or why it failed."""
    import e5_defect_crop as DC
    from PIL import Image, ImageOps
    src, dst = job
    img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    W, H = img.size
    m = Image.open(Path(src).parents[2] / "Masks" / "Anomaly" / (Path(src).stem + ".png")).convert("L")
    if m.size != (W, H):
        return "size"
    small = np.asarray(img.resize((max(1, W // 4), max(1, H // 4)), Image.BILINEAR), dtype=np.float64) / 255
    b = DC.object_box(small)
    if b is None:
        return "no box"
    s = W / small.shape[1]
    x0, y0, x1, y1 = (v * s for v in b)
    side = min(max(x1 - x0, y1 - y0) * (1 + 2 * DC.MARGIN), W, H)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, y0 = float(np.clip(cx - side / 2, 0, W - side)), float(np.clip(cy - side / 2, 0, H - side))
    out = np.asarray(m.crop((round(x0), round(y0), round(x0 + side), round(y0 + side)))
                     .resize((DC.SIZE, DC.SIZE), Image.NEAREST)) > 0
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out.astype(np.uint8) * 255).save(dst)
    return float(out.mean())


def visa_masks(only=None):
    """VisA's defect masks, cropped like visa256crop's copies -> out_s45/visa_masks/<object>/1/<n>.png. `only` limits
    them to those cropped photos (the smoke test's)."""
    import e5_defect_crop as DC
    idx = DC.originals()
    rows = [r for r in idx[idx.label == 1].itertuples() if only is None or str(VISA / r.path) in only]
    jobs = [(r.src, visa_mask(str(VISA / r.path))) for r in rows]
    with Pool(S1.cpus()) as pool:
        res = pool.map(_visa_mask, jobs, chunksize=8)
    bad = [x for x in res if isinstance(x, str)]
    share = np.array([x for x in res if not isinstance(x, str)])
    print(f"VisA masks: {len(share)} of {len(jobs)}; defect share of the crop median {np.median(share):.4f}; failed: "
          f"{pd.Series(bad).value_counts().to_dict()}", flush=True)
    if bad:
        raise SystemExit("some VisA masks could not be cropped like their photos")


def make(task):
    t0 = time.time()
    synth_set(f"s{task}", synth_bases(task), task)
    if task == 0:
        R = reserved()
        synth_set("held", [(f, "leaf") for f in R["held_leaf"]] + [(f, "coil") for f in R["held_coil"]], 100)
        coil = R["probe_coil"]
        synth_set("probe", [(f, "coil") for f in coil[len(coil) // 2:]], 200, fills=coil)
        if QUICK:
            visa_masks({p["abnormal"] for p in mix3_data(0) + held_sets()["visa"] if p["source"] == 1})
        else:
            visa_masks()
    manifest("make", task, seconds=round(time.time() - t0))


# ---------------------------------------------------------------- training
def draw_crop(img, mask=None):
    """The paper's random resized crop (scale 0.5-1) and flip, as parameters. With a mask, the crop is redrawn (at
    most CROP_TRIES tries) until it keeps at least KEEP of the lesion."""
    from torchvision import transforms
    m = None if mask is None else np.asarray(mask) > 127
    for _ in range(CROP_TRIES):
        box = transforms.RandomResizedCrop.get_params(img, scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3))
        if m is None or m[box[0]:box[0] + box[2], box[1]:box[1] + box[3]].sum() >= KEEP * m.sum():
            break
    return box, bool(torch.rand(1) < 0.5)


def augment(img, box, flip):
    from torchvision.transforms import functional as TF
    x = TF.resized_crop(img, *box, [224, 224])
    if flip:
        x = TF.hflip(x)
    return TF.normalize(TF.to_tensor(x), P.MEAN, P.STD)


def patch_targets(mask, box, flip):
    """The 14 x 14 token targets of a lesion mask under a crop: 1 where the 16-px patch is at least PATCH_POS lesion."""
    from torchvision.transforms import functional as TF
    m = TF.resized_crop(mask, *box, [224, 224], interpolation=TF.InterpolationMode.NEAREST)
    if flip:
        m = TF.hflip(m)
    m = np.asarray(m) > 127
    return torch.from_numpy((m.reshape(GRID, 16, GRID, 16).mean((1, 3)) >= PATCH_POS).reshape(-1).astype(np.float32))


class PairSet(torch.utils.data.Dataset):
    """Pairs under one shared crop and flip, so a synthetic pair still differs only in its lesion. Items are (normal,
    abnormal, the abnormal image's patch targets, source), with targets of -1 when the pair has no mask."""

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        from PIL import Image
        p = self.pairs[i]
        a, b = (Image.open(p[k]).convert("RGB") for k in ("normal", "abnormal"))
        if b.size != a.size:
            b = b.resize(a.size, Image.BILINEAR)
        m = Image.open(p["mask"]).convert("L") if p["mask"] else None
        box, flip = draw_crop(a, m)
        tgt = patch_targets(m, box, flip) if m is not None else torch.full((GRID * GRID,), -1.0)
        return augment(a, box, flip), augment(b, box, flip), tgt, p["source"]


class SingleSet(torch.utils.data.Dataset):
    """unpaired_synth's images, each with its own crop and flip (a lesioned image's crop keeps half its lesion). The
    clean images come first, then their lesioned copies in the same order."""

    def __init__(self, pairs):
        self.items = ([(p["normal"], 0.0, None) for p in pairs] + [(p["abnormal"], 1.0, p["mask"]) for p in pairs])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        path, y, mask = self.items[i]
        img = Image.open(path).convert("RGB")
        box, flip = draw_crop(img, Image.open(mask).convert("L") if mask else None)
        return augment(img, box, flip), y


class SplitBatches(torch.utils.data.Sampler):
    """unpaired_synth's batches, in which an image and its lesioned copy never share a batch.

    Each epoch the base images are split at random into halves A and B. The first half of the batches holds A's clean
    images and B's lesioned ones, and the second half the others, shuffled within each half. Item i < n is base image
    i clean, and n + i its lesioned copy."""

    def __init__(self, n, batch, seed):
        self.n, self.batch, self.seed, self.epoch = n, batch, seed, 0

    def __len__(self):
        return 2 * (self.n // self.batch)

    def __iter__(self):
        g = torch.Generator().manual_seed(1000 * self.seed + self.epoch)
        self.epoch += 1
        perm = torch.randperm(self.n, generator=g)
        a, b = perm[:self.n // 2], perm[self.n // 2:]
        for half in (torch.cat([a, b + self.n]), torch.cat([a + self.n, b])):
            half = half[torch.randperm(len(half), generator=g)]
            for k in range(len(half) // self.batch):
                yield half[k * self.batch:(k + 1) * self.batch].tolist()


class Heads(nn.Module):
    """The scorer s(x) on the CLS token, and optionally SP's patch head on the patch tokens and the third test's
    source head."""

    def __init__(self, patch=False, source=False):
        super().__init__()
        self.score = nn.Linear(384, 1)
        self.patch = nn.Linear(384, 1) if patch else None
        self.source = nn.Sequential(nn.Linear(384, 384), nn.ReLU(), nn.Linear(384, 3)) if source else None


class Reverse(torch.autograd.Function):
    """Gradient reversal: the identity forward, and the gradient times -lam backward."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def dann(p):
    """The DANN weight at training progress p in [0, 1] (Ganin & Lempitsky 2015)."""
    return 2 / (1 + math.exp(-10 * p)) - 1


@functools.lru_cache(None)
def gate_choice():
    """The objective the gate carries to the third test. It is "pairs_patch" (the pair loss with SP's patch loss) or
    "pairs" (the pair loss alone, which P1 and P2 share). The smoke test has no gate and uses "pairs_patch"."""
    f = OUT / "gate.json"
    if f.exists():
        return json.load(open(f))["objective"]
    if QUICK:
        return "pairs_patch"
    raise SystemExit("the third test trains only after the gate has run")


def objective(fam):
    """(loss, patch head, source head, adversarial) of a condition."""
    if fam in ("pairs_species_bce", "unpaired_synth"):
        return "bce", False, False, False
    if fam in ("pairs_species", "pairs_synth"):
        return "pair", False, False, False
    if fam == "pairs_synth_patch":
        return "pair", True, False, False
    return "pair", gate_choice() == "pairs_patch", True, fam == "mix3_adv"


def step_loss(m, heads, batch, kind, lam, dev):
    """One batch's loss, with (correct, count) for the log. A pair counts as correct when s(abnormal) > s(normal), and
    a single image when the sign of s matches its label."""
    if len(batch) == 2:                                          # unpaired_synth
        x, y = batch
        s = heads.score(m.forward_features(x.to(dev))[:, 0]).squeeze(1)
        y = y.float().to(dev)
        return F.binary_cross_entropy_with_logits(s, y), int(((s > 0).float() == y).sum()), len(y)
    xa, xb, tgt, src = batch
    nb = len(xa)
    f = m.forward_features(torch.cat([xa, xb]).to(dev))          # after the final norm
    cls, tok = f[:, 0], f[:, m.num_prefix_tokens:]
    s = heads.score(cls).squeeze(1)
    if kind == "pair":
        loss = F.softplus(s[:nb] - s[nb:]).mean()                 # -log sigmoid(s(abnormal) - s(normal))
    else:
        loss = F.binary_cross_entropy_with_logits(s, torch.cat([torch.zeros(nb), torch.ones(nb)]).to(dev))
    if heads.patch is not None:
        tgt = tgt.to(dev)
        t = torch.cat([torch.where(tgt >= 0, torch.zeros_like(tgt), tgt), tgt])   # a normal image: all negative
        ok = t >= 0
        if ok.any():
            loss = loss + PATCH_W * F.binary_cross_entropy_with_logits(heads.patch(tok).squeeze(-1)[ok], t[ok])
    if heads.source is not None:
        loss = loss + F.cross_entropy(heads.source(Reverse.apply(cls, lam)), torch.cat([src, src]).to(dev))
    return loss, int((s[nb:] > s[:nb]).sum()), nb


@torch.no_grad()
def scores(m, heads, paths, dev):
    """s(x) of each image at the evaluation crop (resize 224, centre crop)."""
    out = []
    for x, _ in torch.utils.data.DataLoader(P.ImageList([(p, 0) for p in paths], P.EVAL_TF), batch_size=128,
                                            num_workers=min(6, S1.cpus())):
        out.append(heads.score(m.forward_features(x.to(dev))[:, 0]).squeeze(1).cpu())
    return torch.cat(out).numpy()


def learnt_check(m, heads, fam, dev):
    """Held-out pair accuracy per source and pooled, the share of held-out pairs whose abnormal image scores above its
    normal one."""
    m.eval(), heads.eval()
    out, hits, n = {}, 0, 0
    for name, pairs in held_pairs(fam).items():
        sa = scores(m, heads, [p["normal"] for p in pairs], dev)
        sb = scores(m, heads, [p["abnormal"] for p in pairs], dev)
        out[name] = float((sb > sa).mean())
        hits, n = hits + int((sb > sa).sum()), n + len(pairs)
    out["all"] = hits / n
    print("held-out pair accuracy:", {k: round(v, 3) for k, v in out.items()}, flush=True)
    return out


def train(task):
    fam, seed = RUNS[task]
    ck = CKPT / f"{fam}_s{seed}.pt"
    if ck.exists():
        print("exists:", ck); return
    t0 = time.time()
    kind, patch, source, adv = objective(fam)
    P.seed_all(seed)
    pairs = train_data(fam, seed)
    workers = min(6, S1.cpus())
    if fam == "unpaired_synth":
        dl = torch.utils.data.DataLoader(SingleSet(pairs), batch_sampler=SplitBatches(len(pairs), 2 * BATCH_PAIRS, seed),
                                         num_workers=workers, persistent_workers=True)
    else:
        dl = torch.utils.data.DataLoader(PairSet(pairs), batch_size=BATCH_PAIRS, shuffle=True, drop_last=True,
                                         num_workers=workers, persistent_workers=True)
    assert QUICK or (len(pairs) == 2700 and len(dl) == 84), (len(pairs), len(dl))
    print(f"{fam} seed {seed}: {len(pairs)} pairs, objective {kind}, patch head {patch}, source head {source}, "
          f"adversarial {adv}", flush=True)
    dev = P.device()
    m = S1.imagenet_model(B)
    trainable = tuple(f"blocks.{i}." for i in Q.VIT_TRAINED) + ("norm.",)
    for n_, p in m.named_parameters():
        p.requires_grad = n_.startswith(trainable)
    heads = Heads(patch, source)
    m.to(dev), heads.to(dev)
    params = [p for p in m.parameters() if p.requires_grad] + list(heads.parameters())
    opt = torch.optim.SGD(params, lr=P.LR, momentum=0.9, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    log, done = [], 0
    while done < STEPS:
        m.train(), heads.train()
        t1, tot, hit, n = time.time(), 0.0, 0, 0
        for batch in dl:
            if done == STEPS:
                break
            loss, h_, n_ = step_loss(m, heads, batch, kind, dann(done / STEPS) if adv else 0.0, dev)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(), sched.step()
            done += 1
            tot, hit, n = tot + loss.item() * n_, hit + h_, n + n_
        log.append(dict(epoch=len(log) + 1, loss=tot / max(n, 1), train_acc=hit / max(n, 1), sec=round(time.time() - t1)))
        print(log[-1], flush=True)
    held = learnt_check(m, heads, fam, dev)
    for d in (ck.parent, heads_path(ck.stem).parent):
        d.mkdir(parents=True, exist_ok=True)
    torch.save(heads.state_dict(), heads_path(ck.stem))
    json.dump(dict(backbone=B, source=fam, seed=seed, n=2 * len(pairs), pairs=len(pairs), steps=STEPS, lr=P.LR,
                   objective=dict(loss=kind, patch=patch, source=source, adversarial=adv), log=log,
                   heldout_pair_acc=held, sec=round(time.time() - t0)), open(ck.with_suffix(".json"), "w"), indent=1)
    tmp = ck.with_suffix(".tmp")
    torch.save({k: v.cpu() for k, v in m.state_dict().items() if not k.startswith("head.")}, tmp)   # as S1.train
    os.replace(tmp, ck)
    manifest("train", task, network=ck.stem, heldout_pair_acc=held, seconds=round(time.time() - t0))


def check():
    """Check that under the training augmentation a synthetic pair differs only inside its lesion (dilated by the
    resampling's reach), and that the patch targets match the mask. Saves a contact sheet of crops with their
    positive patches."""
    from PIL import Image, ImageDraw
    from scipy import ndimage
    from torchvision.transforms import functional as TF
    rows = synth_index("s0").head(16)
    tiles, worst = [], 0.0
    for i, r in enumerate(rows.itertuples()):
        torch.manual_seed(i)
        a, b, m = Image.open(r.base).convert("RGB"), Image.open(r.lesioned).convert("RGB"), Image.open(r.mask).convert("L")
        box, flip = draw_crop(a, m)
        xa, xb, tgt = augment(a, box, flip), augment(b, box, flip), patch_targets(m, box, flip)
        mc = TF.resized_crop(m, *box, [224, 224], interpolation=TF.InterpolationMode.NEAREST)
        mc = np.asarray(TF.hflip(mc) if flip else mc) > 127
        diff = (xa - xb).abs().sum(0).numpy()
        reach = max(2, math.ceil(224 / min(box[2], box[3])) + 1)   # bilinear upsampling spreads a pixel this far
        outside = diff[~ndimage.binary_dilation(mc, iterations=reach)]
        worst = max(worst, float(outside.max()) if outside.size else 0.0)
        frac = mc.reshape(GRID, 16, GRID, 16).mean((1, 3)).reshape(-1)
        assert np.array_equal(tgt.numpy() == 1, frac >= PATCH_POS), "patch targets differ from the mask"
        img = Image.fromarray((np.clip(xb.numpy().transpose(1, 2, 0) * P.STD + P.MEAN, 0, 1) * 255).astype(np.uint8))
        d = ImageDraw.Draw(img)
        for j in np.flatnonzero(tgt.numpy() == 1):
            y, x = divmod(int(j), GRID)
            d.rectangle([16 * x, 16 * y, 16 * x + 15, 16 * y + 15], outline=(255, 0, 0))
        tiles.append(img)
    out = Image.new("RGB", (224 * 4, 224 * ((len(tiles) + 3) // 4)))
    for i, t in enumerate(tiles):
        out.paste(t, (224 * (i % 4), 224 * (i // 4)))
    (OUT / "sheets").mkdir(parents=True, exist_ok=True)
    out.save(OUT / "sheets" / "patches.jpg", quality=90)
    print(f"pair check: largest difference outside the dilated lesion {worst:.2e} (normalized units)", flush=True)
    assert worst < 1e-5, "an augmented synthetic pair differs outside its lesion"


# ---------------------------------------------------------------- features
def net_build(c):
    """A network as S1.build makes it. The smoke test's own networks load from its checkpoint folder."""
    if QUICK and c != "imagenet" and not c.startswith(BASE):
        m = S1.imagenet_model(B)
        missing, unexpected = m.load_state_dict(torch.load(CKPT / f"{c}.pt", map_location="cpu"), strict=False)
        assert not unexpected and set(missing) == S1.head_keys(B), (missing, unexpected)
        return m
    return S1.build(B, c)


def heads_of(c, dev):
    sd = torch.load(heads_path(c), map_location="cpu")
    h = Heads(patch="patch.weight" in sd, source="source.0.weight" in sd)
    h.load_state_dict(sd)
    return h.to(dev).eval()


def norm(dev):
    return (torch.tensor(v).view(1, 3, 1, 1).to(dev) for v in S1.norm_stats(B))


@torch.no_grad()
def quick_tests(m, c, dev):
    """Smoke test: the readouts of the first N_QUICK images of each test, into out_s45/quick (the pipeline's folders
    are not touched)."""
    mean, std = norm(dev)
    for t in TESTS:
        tab, ims = S1.load_target(t) if t in S1.TARGETS else S4.load_isic(t)
        x = torch.from_numpy(np.array(ims[:N_QUICK])).to(dev).float().div_(255)
        f = S1.stage_features(B, m, (x - mean) / std)
        assert set(f) == {"b6", "b9", "b12", "cls"} and all(v.shape == (N_QUICK, 384) for v in f.values())
        (OUT / "feats" / t).mkdir(parents=True, exist_ok=True)
        np.savez(OUT / "feats" / t / f"feats_{c}.npz", **{k: v.cpu().numpy().astype(np.float16) for k, v in f.items()},
                 y=tab["y"].values[:N_QUICK])


@torch.no_grad()
def tokens(m, c, dev):
    """The ISIC 2018 Task 1 images' final-block patch tokens after the final norm, as in the robust-PCA test
    -> out_s45/tokens/<c>.npy, float16 (N, 196, 384). diagnose deletes them once localization is scored."""
    dst = OUT / "tokens" / f"{c}.npy"
    if dst.exists() or (OUT / "loc" / f"{c}.npz").exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tab, ims = S4.load_isic("isic_attr")
    mean, std = norm(dev)
    tmp = dst.with_suffix(".tmp.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(len(tab), GRID * GRID, 384))
    for i in range(0, len(tab), 128):
        x = torch.from_numpy(np.array(ims[i:i + 128])).to(dev).float().div_(255)
        arr[i:i + len(x)] = m.forward_features((x - mean) / std)[:, m.num_prefix_tokens:].float().cpu().numpy()
    arr.flush()
    del arr
    os.replace(tmp, dst)


@torch.no_grad()
def probe_features(m, c, dev):
    """The probe set's readouts (S1.stage_features) at the evaluation crop -> out_s45/probe/<c>.npz."""
    dst = OUT / "probe" / f"{c}.npz"
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    items = probe_items()
    feats = {}
    for x, _ in torch.utils.data.DataLoader(P.ImageList([(p, 0) for p, _, _ in items], P.EVAL_TF), batch_size=128,
                                            num_workers=min(6, S1.cpus())):
        for k, v in S1.stage_features(B, m, x.to(dev)).items():
            feats.setdefault(k, []).append(v.cpu().numpy().astype(np.float16))
    np.savez(dst.with_suffix(".tmp.npz"), **{k: np.concatenate(v) for k, v in feats.items()},
             source=np.array([s for _, s, _ in items]), abnormal=np.array([a for _, _, a in items]))
    os.replace(dst.with_suffix(".tmp.npz"), dst)


@torch.no_grad()
def attn_features(m, c, dev):
    """The exploratory SP-attn readout on HAM10000 and ISIC -> out_s45/attn/<test>/feats_<c>.npz (key "attn"). It
    pools the final-block patch tokens (after the final norm), weighted by the softmax of the patch head's scores."""
    heads = heads_of(c, dev)
    mean, std = norm(dev)
    for t in ("ham", "isic_test"):
        dst = OUT / "attn" / t / f"feats_{c}.npz"
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tab, ims = S1.load_target(t) if t in S1.TARGETS else S4.load_isic(t)
        n = N_QUICK if QUICK else len(tab)
        out = []
        for i in range(0, n, 256):
            x = torch.from_numpy(np.array(ims[i:min(i + 256, n)])).to(dev).float().div_(255)
            tok = m.forward_features((x - mean) / std)[:, m.num_prefix_tokens:]
            w = torch.softmax(heads.patch(tok).squeeze(-1), 1)
            out.append((w.unsqueeze(-1) * tok).sum(1).cpu().numpy().astype(np.float16))
        np.savez(dst.with_suffix(".tmp.npz"), attn=np.concatenate(out), y=tab["y"].values[:n])
        os.replace(dst.with_suffix(".tmp.npz"), dst)


def features(task):
    c = NETS[task]
    t0 = time.time()
    if not QUICK:
        for t in TESTS:                                    # skips the files that exist (B0's and B1's, mostly)
            S4.extract_conditions(B, t, [c])
    dev = P.device()
    m = net_build(c).eval().to(dev)
    if QUICK:
        quick_tests(m, c, dev)
    tokens(m, c, dev)
    probe_features(m, c, dev)
    if family(c) == "pairs_synth_patch":
        attn_features(m, c, dev)
    print(f"{c}: features in {time.time() - t0:.0f}s", flush=True)
    manifest("features", task, network=c, seconds=round(time.time() - t0))


# ---------------------------------------------------------------- few-shot evaluation
def preds_path(t, c):
    """Where network c's predictions on target t go. HAM10000 and DDR use the pipeline's folder, the others out_s45."""
    return (S1.OUT / B if t in S1.TARGETS else OUT / B) / t / f"preds_{c}.npz"


def other_splits(t):
    """HyperKvasir, PAD-UFES-20 and HIBA: labels, groups, classes and S1.make_splits with their k values."""
    tab = pd.read_csv(S4.OUT / "cache" / f"{t}_index.csv")
    y, g = tab["y"].values, tab["group"].values
    ks, S1.KS = S1.KS, OTHER_KS
    try:
        splits = S1.make_splits(y, g, int(y.max()) + 1)
    finally:
        S1.KS = ks
    return y, g, int(y.max()) + 1, splits


def evaluate_other(t, cs):
    """S4.evaluate_conditions for the other targets, keeping the validation accuracies, into out_s45."""
    from joblib import Parallel, delayed
    y, _, nc, splits = other_splits(t)
    splits = {sp: dict(splits[sp], shots={k: v[:DRAWS] for k, v in splits[sp]["shots"].items()}) for sp in SPLITS}
    todo = [c for c in cs if not preds_path(t, c).exists()]
    jobs = [(c, r) for c in todo for r in S1.readouts(B)]
    out = Parallel(n_jobs=S1.cpus(), verbose=1)(
        delayed(S1._eval_job)(S4.target_dir(B, t) / f"feats_{c}.npz", B, c, r, splits, y, nc, OTHER_KS) for c, r in jobs)
    per = {}
    for c, res in out:
        per.setdefault(c, {}).update(res)
    for c, res in per.items():
        for sp in SPLITS:
            res[f"split|{sp}|te"] = splits[sp]["te"]
        E10.save(preds_path(t, c), res)


def attn_evaluate(c):
    """Exploratory: SP with the attn readout beside the standard ones -> out_s45/attn/ext_<c>.npz.

    Per split and k, the readout with the best mean HAM10000 validation accuracy wins, with ties going to the standard
    ones. ISIC is predicted from the HAM10000 shots at that readout. The keys are best|split|k (the predictions) and
    attn|split|k (whether attn won)."""
    dst = OUT / "attn" / f"ext_{c}.npz"
    if dst.exists():
        return
    tab, _ = S1.load_target("ham")
    y, g = tab["y"].values, tab["group"].values
    splits = S1.make_splits(y, g, 7)
    fa = OUT / "attn" / "ham" / f"feats_{c}.npz"
    _, res = S1._eval_job(fa, B, c, "attn", {sp: splits[sp] for sp in SPLITS}, y, 7, S1.KS)
    std = np.load(S1.OUT / B / "ham" / f"preds_{c}.npz")
    E = np.load(S4.OUT / B / "isic_test" / f"ext_{c}.npz")
    Xh = np.load(fa)["attn"].astype(np.float32)
    Xe = np.load(OUT / "attn" / "isic_test" / f"feats_{c}.npz")["attn"].astype(np.float32)
    out = {}
    for sp in SPLITS:
        for k in S1.KS:
            cand = {r: std[f"{r}|{sp}|{k}|val_ba"].mean() for r in S1.readouts(B)}
            cand["attn"] = res[f"attn|{sp}|{k}|val_ba"].mean()
            won = max(cand, key=cand.get) == "attn"
            out[f"attn|{sp}|{k}"] = np.array(won)
            out[f"best|{sp}|{k}"] = (np.stack([S4._fit_predict_ext(Xh[idx], y[idx], Xe, None)
                                               for idx in splits[sp]["shots"][k]]) if won else E[f"best|{sp}|{k}"])
    E10.save(dst, out)


def evaluate(task):
    c = NETS[task]
    t0 = time.time()
    if QUICK:                                              # the existing networks only, on the other targets
        evaluate_other_quick(c)
    else:
        for t in ("ham", "ddr"):
            S4.evaluate_conditions(B, t, [c])
        S4.external(B, [c])                                # ISIC 2018 test, from the HAM10000 shots
        for t in OTHER:
            evaluate_other(t, [c])
        if family(c) == "pairs_synth_patch":
            attn_evaluate(c)
    manifest("evaluate", task, network=c, seconds=round(time.time() - t0))


def evaluate_other_quick(c):
    for t in OTHER:
        if (S4.target_dir(B, t) / f"feats_{c}.npz").exists():
            evaluate_other(t, [c])
        else:
            print(f"{t} {c}: no features yet (the full run extracts them)", flush=True)


# ---------------------------------------------------------------- diagnostics
def fit_half(n):
    """The robust-PCA test's fixed half of the Task 1 images, used for channel statistics. The other half is scored."""
    fit = np.zeros(n, bool)
    fit[np.random.RandomState(1).permutation(n)[: n // 2]] = True
    return fit


def pcp(M, lam=None):
    """Principal component pursuit by inexact ALM, min ||L||_* + lam ||S||_1 subject to L + S = M. As in our earlier
    robust-PCA test."""
    lam = 1 / np.sqrt(max(M.shape)) if lam is None else lam
    norm2 = np.linalg.norm(M, 2)
    Y = M / max(norm2, np.abs(M).max() / lam)
    mu, S, fro = 1.25 / norm2, np.zeros_like(M), np.linalg.norm(M)
    mu_bar = mu * 1e7
    for it in range(1, MAX_ITER + 1):
        U, sig, Vt = np.linalg.svd(M - S + Y / mu, full_matrices=False)
        r = int((sig > 1 / mu).sum())
        L = (U[:, :r] * (sig[:r] - 1 / mu)) @ Vt[:r]
        T = M - L + Y / mu
        S = np.sign(T) * np.maximum(np.abs(T) - lam / mu, 0)
        Z = M - L - S
        Y = Y + mu * Z
        mu = min(mu * RHO, mu_bar)
        if np.linalg.norm(Z) / fro < TOL:
            break
    return L, S, it


def upsample(scores):
    """(N, 196) patch scores -> (N, 224, 224), bilinear."""
    t = torch.as_tensor(np.asarray(scores, np.float32)).view(-1, 1, GRID, GRID)
    return F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)[:, 0].numpy()


def auroc(labels, scores):
    from sklearn.metrics import roc_auc_score
    labels = np.asarray(labels, bool).reshape(-1)
    return float(roc_auc_score(labels, np.asarray(scores).reshape(-1))) if 0 < labels.sum() < labels.size else np.nan


def unpack(bits):
    return np.unpackbits(bits, axis=-1)[..., :224 * 224].reshape(*bits.shape[:-1], 224, 224).astype(bool)


def score_half():
    """The Task 1 images whose localization is scored (only the first N_LOC_QUICK in the smoke test)."""
    idx = np.flatnonzero(~fit_half(len(S4.load_isic("isic_attr")[0])))
    return idx[:N_LOC_QUICK] if QUICK else idx


def _loc_job(path, idx, mu, sd, w, b):
    T = np.load(path, mmap_mode="r")
    masks = np.load(S41 / "masks_task1.npz")["mask"]
    out = {"i": [], "v1": [], "patch": []}
    for i in idx:
        tok = np.asarray(T[i], np.float64)
        _, S, _ = pcp((tok - mu) / sd)
        mask = unpack(masks[i])
        out["i"].append(i)
        out["v1"].append(auroc(mask, upsample(np.linalg.norm(S, axis=1)[None])[0]))
        out["patch"].append(np.nan if w is None else auroc(mask, upsample((tok @ w + b)[None])[0]))
    return {k: np.array(v) for k, v in out.items()}


def localize(c):
    """Pixel AUROC per Task 1 score-half image -> out_s45/loc/<c>.npz.

    Two maps are scored, both upsampled bilinearly to 224 px. V1 is the robust-PCA test's map (the norm of each
    token's sparse part) on the network's tokens, z-scored with their own fit-half statistics. The other is SP's patch
    head. The tokens are deleted afterwards, except in the smoke test."""
    from joblib import Parallel, delayed
    dst, src = OUT / "loc" / f"{c}.npz", OUT / "tokens" / f"{c}.npy"
    if dst.exists():
        return
    T = np.load(src, mmap_mode="r")
    X = np.asarray(T[np.flatnonzero(fit_half(len(T)))], np.float64).reshape(-1, T.shape[2])
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-8] = 1.0
    del X
    w = b = None
    if family(c) == "pairs_synth_patch":
        h = torch.load(heads_path(c), map_location="cpu")
        w, b = h["patch.weight"].numpy()[0].astype(np.float64), float(h["patch.bias"])
    idx = score_half()
    parts = Parallel(n_jobs=S1.cpus())(delayed(_loc_job)(str(src), ch, mu, sd, w, b)
                                       for ch in np.array_split(idx, min(len(idx), 8 * S1.cpus())))
    res = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst.with_suffix(".tmp.npz"), **res)
    os.replace(dst.with_suffix(".tmp.npz"), dst)
    print(f"{c}: V1 mean AUROC {np.nanmean(res['v1']):.4f}" + ("" if w is None else
                                                                f", patch head {np.nanmean(res['patch']):.4f}"), flush=True)
    if QUICK and c == "imagenet":                          # compare with the robust-PCA test's own scores
        ref = np.load(S41 / "scores_task1.npz")
        old = dict(zip(ref["i"], ref["auc1"]))
        gap = max(abs(old[i] - v) for i, v in zip(res["i"], res["v1"]))
        print(f"V1 against the robust-PCA test's scores of the same {len(res['i'])} images: largest gap {gap:.2e}",
              flush=True)
        assert gap < 1e-3, "V1 does not reproduce the robust-PCA test"
    if not QUICK:
        src.unlink()


def decode(c):
    """Source decodability, the five-fold balanced accuracy of a linear probe for each image's source, at the CLS token
    and at the last block's mean patch token. Folds are stratified by source, features are z-scored on the training
    folds, and the probe is logistic regression with C = 0.1."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    dst = OUT / "decode" / f"{c}.json"
    if dst.exists():
        return
    Z = np.load(OUT / "probe" / f"{c}.npz")
    y, out = Z["source"], {}
    for r in ("cls", "b12"):
        X, accs = Z[r].astype(np.float32), []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
            mu, sd = X[tr].mean(0), X[tr].std(0)
            sd[sd < 1e-8] = 1.0
            p = LogisticRegression(C=Q.C, max_iter=3000).fit((X[tr] - mu) / sd, y[tr]).predict((X[te] - mu) / sd)
            accs.append(np.mean([(p[y[te] == k] == k).mean() for k in np.unique(y[te])]))
        out[r] = 100 * float(np.mean(accs))
    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dst, "w"))
    print(f"{c}: source decodability {out}", flush=True)


def standin(c):
    """Linear-map retention, as the paper's stand-in (e9_linear_stand_in.standin) -> out_s45/standin/<c>.npz.

    A ridge map from ImageNet's all-layer features to the network's is fit on the unlabeled HAM10000 images, each side
    standardized with its HAM10000 statistics. The stand-in's features then go through the few-shot protocol on
    HAM10000 and ISIC. The file holds their correctness under keys <test>|<split>|<k>."""
    dst = OUT / "standin" / f"{c}.npz"
    if dst.exists():
        return
    tests = ("ham", "isic_test")
    raw = {t: E9.load(B, t, "imagenet") for t in tests}
    sx = E9.Std(raw["ham"])
    X = {t: sx(v) for t, v in raw.items()}
    raw = {t: E9.load(B, t, c) for t in tests}
    sy = E9.Std(raw["ham"])
    Y = {t: sy(v) for t, v in raw.items()}
    A, c0, alpha = E9.ridge(X["ham"], Y["ham"])
    T, ye = E9.skin_tests()
    C, _ = E9.evaluate({"standin": {t: X[t] @ A + c0 for t in tests}}, B, T, ye)
    dst.parent.mkdir(parents=True, exist_ok=True)
    E10.save(dst, {k.replace("|standin|", "|"): v for k, v in C.items()} |
             dict(alpha=np.array(alpha), r2_isic=np.array(E9.r2(Y["isic_test"], X["isic_test"] @ A + c0))))
    print(f"{c}: stand-in fitted (ridge {alpha:g})", flush=True)


def diagnose(task):
    c = NETS[task]
    t0 = time.time()
    localize(c)
    decode(c)
    if c != "imagenet" and (not QUICK or c.startswith(BASE)):
        standin(c)
    manifest("diagnose", task, network=c, seconds=round(time.time() - t0))


# ---------------------------------------------------------------- scoring
def boot_p(bs):
    """Two-sided bootstrap p-value, the smallest alpha at which the percentile interval excludes 0 (as
    e2_robustness.boot_p)."""
    return float(min(1.0, 2 * max(min((bs <= 0).mean(), (bs >= 0).mean()), 0.5 / len(bs))))


def holm(p):
    """Holm-adjusted p-values (e2_robustness.holm)."""
    p = np.asarray(p, float)
    o, m = np.argsort(p), len(p)
    adj = np.empty(m)
    adj[o] = np.minimum(np.maximum.accumulate((m - np.arange(m)) * p[o]), 1)
    return adj


def t_interval(x):
    """The mean of per-seed values with its 95% t-interval (df = n - 1) and two-sided p."""
    from scipy import stats
    x = np.asarray(x, float)
    n, mean = len(x), float(x.mean())
    if n < 2:
        return mean, np.nan, np.nan, np.nan
    se = float(x.std(ddof=1) / np.sqrt(n))
    if se == 0:
        return mean, mean, mean, float(mean == 0)
    q = stats.t.ppf(0.975, n - 1)
    return mean, mean - q * se, mean + q * se, float(2 * stats.t.sf(abs(mean) / se, n - 1))


class Boot45(E10.Boot40):
    def dist(self, diff):
        """Boot40.gain's point estimate, with its bootstrap distribution and the seed x split cells."""
        nt, ns = len(self.seeds), len(self.splits)
        G = np.empty((self.n_boot, nt, ns))
        cells = []
        for ti, t in enumerate(self.seeds):
            for si, sp in enumerate(self.splits):
                d = diff(t, sp)
                cells.append(self.ba(d, sp))
                with np.errstate(invalid="ignore", divide="ignore"):
                    G[:, ti, si] = np.nanmean(np.einsum("in,nc->ic", self.w[sp] * d, self.Yoh[sp]) / self.den[sp], 1)
        gm = G.mean((1, 2), keepdims=True)
        rt, cs = G.mean(2, keepdims=True) - gm, G.mean(1, keepdims=True) - gm
        if nt > 1:
            G = G + (np.sqrt(nt / (nt - 1)) - 1) * rt
        if ns > 1:
            G = G + (np.sqrt(ns / (ns - 1)) - 1) * cs
        return float(np.mean(cells)), np.einsum("it,its,is->i", self.Ct, G, self.Cs) / (nt * ns), cells


def test_data(t):
    """Test labels and resampling units per split, the number of classes and the k values. ISIC uses the same 1,512
    images for every split, resampled by lesion (E10.isic_units)."""
    if t == "isic_test":
        ye = S4.load_isic("isic_test")[0]["y"].values
        u, _ = E10.isic_units()
        return {sp: ye for sp in SPLITS}, {sp: u for sp in SPLITS}, 7, S1.KS
    y, g, nc, splits = E10.target(t) if t in S1.TARGETS else other_splits(t)
    return ({sp: y[splits[sp]["te"]] for sp in SPLITS}, {sp: pd.factorize(g[splits[sp]["te"]])[0] for sp in SPLITS},
            nc, S1.KS if t in S1.TARGETS else OTHER_KS)


def done(t, c):
    return (S4.OUT / B / "isic_test" / f"ext_{c}.npz" if t == "isic_test" else preds_path(t, c)).exists()


def scored(t, c, yte, ks):
    """Per (split, k), the correctness (draws x test images) at the readout chosen on validation, and its mean
    validation accuracy. ISIC uses S4.external's "best" predictions (at the readout chosen on HAM10000 validation) and
    has no validation of its own."""
    if t == "isic_test":
        E = np.load(S4.OUT / B / "isic_test" / f"ext_{c}.npz")
        return {(sp, k): E[f"best|{sp}|{k}"][:DRAWS] == yte[sp][None] for sp in SPLITS for k in ks}, None
    Z = np.load(preds_path(t, c))
    C, V = {}, {}
    for sp in SPLITS:
        for k in ks:
            assert len(Z[f"split|{sp}|te"]) == len(yte[sp])
            r = max(S1.readouts(B), key=lambda r_: Z[f"{r_}|{sp}|{k}|val_ba"][:DRAWS].mean())
            C[sp, k] = Z[f"{r}|{sp}|{k}|test_pred"][:DRAWS] == yte[sp][None]
            V[sp, k] = float(Z[f"{r}|{sp}|{k}|val_ba"][:DRAWS].mean())
    return C, V


def level(C, yte, nc, ks):
    """Balanced accuracy (%), mean over draws, splits and k."""
    return 100 * np.mean([np.mean([C[sp, k][:, yte[sp] == c].mean() for c in range(nc)]) for sp in SPLITS for k in ks])


class Scores:
    """Every network's few-shot correctness on every test, and the comparisons between families (seed s against
    seed s; B0 has no seeds)."""

    def __init__(self, families):
        self.data = {t: test_data(t) for t in TESTS}
        self.C, self.V, self.boots = {}, {}, {}
        for fam in families:
            for s in self.seeds(fam) or [None]:
                c = cond(fam, s)
                for t in TESTS:
                    if (c, t) not in self.C and done(t, c):
                        yte, _, _, ks = self.data[t]
                        self.C[c, t], self.V[c, t] = scored(t, c, yte, ks)

    @staticmethod
    def seeds(fam):
        return [] if fam == "imagenet" else SCORE_SEEDS

    def has(self, fam, t):
        return all((cond(fam, s), t) in self.C for s in self.seeds(fam) or [None])

    def level(self, fam, t):
        yte, _, nc, ks = self.data[t]
        return float(np.mean([level(self.C[cond(fam, s), t], yte, nc, ks) for s in self.seeds(fam) or [None]]))

    def val(self, fam):
        """Mean validation balanced accuracy (%) over seeds, splits and k, then over the five targets."""
        return 100 * float(np.mean([np.mean([np.mean(list(self.V[cond(fam, s), t].values()))
                                             for s in self.seeds(fam) or [None]]) for t in TARGETS]))

    def boot(self, t, seeds):
        key = (t, tuple(seeds) if seeds else None)
        if key not in self.boots:
            yte, gidx, nc, _ = self.data[t]
            self.boots[key] = Boot45(yte, gidx, nc, seeds=seeds, shared=t == "isic_test")
        return self.boots[key]

    def compare(self, t, a, b, C_a=None):
        """Family a minus family b on test t, in points, mean over k. The reported interval is the wider of the
        bootstrap's percentile interval and the t-interval over seed means, and p is the larger of their p-values.
        C_a replaces a's correctness ({seed: {(split, k): array}}, e.g. a stand-in's)."""
        if C_a is None and not self.has(a, t) or not self.has(b, t):
            return None
        _, _, _, ks = self.data[t]
        seeds = sorted(set(self.seeds(a) + self.seeds(b))) or None
        get_a = (lambda s: C_a[s]) if C_a is not None else (lambda s: self.C[cond(a, s), t])
        diff = lambda s, sp: np.mean([get_a(s)[sp, k].mean(0) - self.C[cond(b, s), t][sp, k].mean(0) for k in ks], 0)
        g, bs, cells = self.boot(t, seeds).dist(diff)
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        per_seed = np.array(cells).reshape(len(seeds or [None]), len(SPLITS)).mean(1)
        _, tlo, thi, tp = t_interval(per_seed)
        wide_t = not np.isnan(tlo) and thi - tlo > hi - lo
        return dict(gain=100 * g, lo=100 * (tlo if wide_t else lo), hi=100 * (thi if wide_t else hi),
                    interval="t" if wide_t else "bootstrap", p=max(boot_p(bs), 0.0 if np.isnan(tp) else tp),
                    p_boot=boot_p(bs), p_t=tp, cells=f"{sum(x > 0 for x in cells)}/{len(cells)}")


def verdict_row(rows, pid):
    return next(r for r in rows if r["prediction"] == pid)


def learnt(fam):
    """Whether every seed's held-out pair accuracy reached LEARNT (None: no record, as for B0 and B1)."""
    if fam in ("imagenet", BASE) or QUICK:
        return None
    acc = []
    for s in SEEDS:
        f = CKPT / f"{fam}_s{s}.json"
        if not f.exists():
            return None
        acc.append(json.load(open(f))["heldout_pair_acc"]["all"])
    return bool(min(acc) >= LEARNT)


def learning_rows():
    rows = []
    for fam, s in RUNS:
        f = CKPT / f"{fam}_s{s}.json"
        if f.exists():
            j = json.load(open(f))
            rows.append(dict(network=CODE[fam], seed=s, train_acc=j["log"][-1]["train_acc"], loss=j["log"][-1]["loss"],
                             **{f"heldout_{k}": v for k, v in j["heldout_pair_acc"].items()}, minutes=j["sec"] / 60))
    return pd.DataFrame(rows)


def loc_file(fam, s):
    """A network's localization scores. The smoke test's SP is its own seed-0 network."""
    if QUICK and fam == "pairs_synth_patch":
        return OUT / "loc" / "pairs_synth_patch_s0.npz"
    return OUT / "loc" / f"{cond(fam, s)}.npz"


@functools.lru_cache(None)
def center_prior():
    """Pixel AUROC per score-half image of a fixed map falling off with distance from the image centre."""
    masks = np.load(S41 / "masks_task1.npz")["mask"]
    yy, xx = np.mgrid[:224, :224]
    m = -np.hypot(yy - 111.5, xx - 111.5)
    idx = score_half()
    return idx, np.array([auroc(unpack(masks[i]), m) for i in idx])


def auc_compare(A, ref):
    """Mean of A (seeds x images) minus ref (images) over seeds and images, in AUROC units. The reported interval is
    the wider of the bootstrap over seeds and images and the t-interval over seed means."""
    ok = ~np.isnan(ref) & ~np.isnan(A).any(0)
    D = A[:, ok] - ref[ok][None]
    rng = np.random.RandomState(0)
    bs = np.array([D[rng.randint(0, len(D), len(D))][:, rng.randint(0, D.shape[1], D.shape[1])].mean()
                   for _ in range(Q.N_BOOT)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    _, tlo, thi, _ = t_interval(D.mean(1))
    wide_t = not np.isnan(tlo) and thi - tlo > hi - lo
    return dict(diff=float(D.mean()), lo=float(tlo if wide_t else lo), hi=float(thi if wide_t else hi))


def diagnostics(sc):
    """Per family: retention (ISIC), source decodability, V1 localization (and SP's patch head), mean over seeds."""
    rows, loc = [], {}
    for fam in ["imagenet", BASE] + FAMILIES:
        seeds = Scores.seeds(fam) or [None]
        files = [loc_file(fam, s) for s in seeds]
        r = dict(network=CODE[fam])
        if all(f.exists() for f in files):
            L = [np.load(f) for f in files]
            loc[fam] = dict(i=L[0]["i"], v1=np.stack([z["v1"] for z in L]), patch=np.stack([z["patch"] for z in L]))
            r["v1_auroc"] = float(np.nanmean(loc[fam]["v1"]))
            if not np.isnan(loc[fam]["patch"]).all():
                r["patch_auroc"] = float(np.nanmean(loc[fam]["patch"]))
        dec = [OUT / "decode" / f"{cond(fam, s)}.json" for s in seeds]
        if all(f.exists() for f in dec):
            D = [json.load(open(f)) for f in dec]
            r["decode_cls"], r["decode_b12"] = (float(np.mean([d[k] for d in D])) for k in ("cls", "b12"))
            r["decode_cls_seeds"] = [d["cls"] for d in D]
        if fam != "imagenet":
            st = retention(sc, fam)
            if st:
                r.update(st)
        rows.append(r)
    return pd.DataFrame(rows), loc


def retention(sc, fam):
    """The stand-in's ISIC gain over B0 and its share of the network's, both mean over seeds."""
    files = {s: OUT / "standin" / f"{cond(fam, s)}.npz" for s in Scores.seeds(fam)}
    if not all(f.exists() for f in files.values()) or not sc.has(fam, "isic_test") or not sc.has("imagenet", "isic_test"):
        return None
    yte, _, nc, ks = sc.data["isic_test"]
    C = {}
    for s, f in files.items():
        Z = np.load(f)
        C[s] = {(sp, k): Z[f"isic_test|{sp}|{k}"][:DRAWS] for sp in SPLITS for k in ks}
    r = sc.compare("isic_test", fam, "imagenet", C_a=C)
    net = sc.compare("isic_test", fam, "imagenet")
    return dict(standin_gain=r["gain"], standin_lo=r["lo"], standin_hi=r["hi"], net_gain=net["gain"],
                net_lo=net["lo"], share=r["gain"] / net["gain"] if net["gain"] else np.nan)


def results():
    """Levels, gains, pre-registered comparisons, diagnostics, the gate's choice and the predictions, from the files
    that exist. The gate and the summary both report these."""
    sc = Scores(["imagenet", BASE] + FAMILIES)
    have = [f for f in ["imagenet", BASE] + FAMILIES if sc.has(f, "isic_test")]
    levels = pd.DataFrame([dict(network=CODE[f], test=LABEL[t], accuracy=sc.level(f, t))
                           for f in have for t in TESTS if sc.has(f, t)])
    gains = []
    for f in [x for x in have if x != "imagenet"]:
        for t in TESTS:
            r = sc.compare(t, f, "imagenet")
            if r:
                gains.append(dict(comparison=f"{CODE[f]} - B0", test=LABEL[t], **r))
        if f != BASE and (r := sc.compare("isic_test", f, BASE)):
            gains.append(dict(comparison=f"{CODE[f]} - B1", test=LABEL["isic_test"], **r))
    diag, loc = diagnostics(sc)
    D = diag.set_index("network")
    cmp_ = lambda t, a, b: sc.compare(t, a, b)
    rows, extra = [], []

    def add(pid, test, claim, r=None, holds=None, value=None, primary=False, note=""):
        rows.append(dict(prediction=pid, test=test, primary=primary, claim=claim, holds=holds, note=note,
                         value=value if value is not None else ("not run" if r is None else
                                                                 f"{r['gain']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]"),
                         p=None if r is None else r["p"]))

    def primaries(test, items):
        """Add a test's primary predictions on ISIC, Holm-corrected over the ones that ran, and return whether each
        holds. items: (id, claim, (a, b), bar)."""
        rs = [cmp_("isic_test", a, b) for _, _, (a, b), _ in items]
        ran = [i for i, r in enumerate(rs) if r]
        adj = dict(zip(ran, holm([rs[i]["p"] for i in ran]))) if ran else {}
        for i, (pid, claim, (a, b), bar) in enumerate(items):
            r = rs[i]
            ok = None
            if r:
                ok = bool(adj[i] < 0.05 and r["gain"] > 0 and r["gain"] >= bar)
                if learnt(a) is False or learnt(b) is False:
                    ok = None
            add(pid, test, claim, r, ok, primary=True, note="" if not r else f"Holm p = {adj[i]:.3g}" +
                ("; not learnt" if learnt(a) is False or learnt(b) is False else ""))
            rows[-1]["p_holm"] = adj.get(i)
        return {pid: rows[-len(items) + j]["holds"] for j, (pid, *_) in enumerate(items)}

    def any_of(pid, test, claim, items):
        """Holds when any comparison in items [(test, a, b)], Holm-corrected over the ones that ran, has p < 0.05 and
        a positive gain."""
        rs = [(t, a, b, cmp_(t, a, b)) for t, a, b in items]
        ran = [x for x in rs if x[3]]
        if not ran:
            return add(pid, test, claim)
        adj = holm([r["p"] for *_, r in ran])
        ok = bool(any(q < 0.05 and r["gain"] > 0 for q, (*_, r) in zip(adj, ran)))
        if any(learnt(a) is False for _, a, _, _ in ran):
            ok = None
        add(pid, test, claim, holds=ok, value="; ".join(f"{CODE[a]} - {CODE[b]} {LABEL[t]} {r['gain']:+.2f} "
                                                        f"[{r['lo']:+.2f}, {r['hi']:+.2f}] Holm p {q:.3g}"
                                                        for (t, a, b, r), q in zip(ran, adj)))

    # ---- test 1: matched pairs
    primaries(1, [("S1a", "P1 beats B1 on ISIC by at least +0.5", ("pairs_species", BASE), 0.5),
                  ("S1b", "P1 beats P1-ctrl on ISIC", ("pairs_species", "pairs_species_bce"), 0.0),
                  ("S1c", "P2 beats B0 on ISIC by at least +2.0", ("pairs_synth", "imagenet"), 2.0),
                  ("S1d", "P2 beats P2-unpaired on ISIC", ("pairs_synth", "unpaired_synth"), 0.0)])
    any_of("S1e", 1, "P1 or P2 beats B0 on DDR or HyperKvasir (Holm over the four)",
           [(t, f, "imagenet") for f in ("pairs_species", "pairs_synth") for t in ("ddr", "hyperkvasir6")])
    shares = {}
    for f in ("pairs_species", "pairs_synth"):
        if CODE[f] in D.index and "share" in D.columns and not pd.isna(D.loc[CODE[f], "share"]):
            shares[f] = (D.loc[CODE[f], "share"], D.loc[CODE[f], "net_lo"] > 0)
    scored_ = {f: v for f, (v, ok) in shares.items() if ok}
    add("S1f", 1, "linear-map retention at least 80% for P1 and for P2 (scored where the ISIC gain's CI is above 0)",
        holds=bool(all(v >= RETAIN for v in scored_.values())) if scored_ else None,
        value="; ".join(f"{CODE[f]} {v:.0%}" + ("" if ok else " (not scored)") for f, (v, ok) in shares.items())
        or "not run")
    for a, b in [("pairs_species_bce", BASE)]:
        if r := cmp_("isic_test", a, b):
            extra.append(dict(comparison=f"{CODE[a]} - {CODE[b]}", test=LABEL["isic_test"], **r))

    # ---- gate 1: which pair objective goes on
    gate = {}
    if all(all(sc.has(f, t) for t in TARGETS) for f in ("pairs_species", "pairs_synth", BASE)):
        v1, v2, vb = sc.val("pairs_species"), sc.val("pairs_synth"), sc.val(BASE)
        gate.update(val_p1=v1, val_p2=v2, val_b1=vb, winner_test1="pairs_synth" if v1 - v2 < GAP else "pairs_species",
                    test1_null=bool(max(v1, v2) <= vb))

    # ---- test 2: pointing with free masks
    h2 = primaries(2, [("S2a", "SP beats P2 on ISIC by at least +0.5", ("pairs_synth_patch", "pairs_synth"), 0.5)])
    ci, cp = center_prior()
    sp_loc, b0_loc = loc.get("pairs_synth_patch"), loc.get("imagenet")
    if sp_loc is not None and b0_loc is not None and not np.isnan(sp_loc["patch"]).all():
        assert np.array_equal(sp_loc["i"], ci) and np.array_equal(b0_loc["i"], ci)
        auc = float(np.nanmean(sp_loc["patch"]))
        vs_v1, vs_c = auc_compare(sp_loc["patch"], b0_loc["v1"][0]), auc_compare(sp_loc["patch"], cp)
        ok = bool(auc >= 0.75 and vs_v1["lo"] > 0 and vs_c["lo"] > 0)
        add("S2b", 2, "SP's patch head localizes Task 1 lesions zero-shot with pixel AUROC >= 0.75, above robust PCA on "
            "ImageNet features and above the center prior", holds=None if learnt("pairs_synth_patch") is False else ok,
            value=f"{auc:.3f}; minus V1 (B0) {vs_v1['diff']:+.3f} [{vs_v1['lo']:+.3f}, {vs_v1['hi']:+.3f}]; minus the "
            f"center prior ({np.nanmean(cp):.3f}) {vs_c['diff']:+.3f} [{vs_c['lo']:+.3f}, {vs_c['hi']:+.3f}]")
    else:
        add("S2b", 2, "SP's patch head localizes Task 1 lesions zero-shot with pixel AUROC >= 0.75, above robust PCA on "
            "ImageNet features and above the center prior")
    s2b = rows[-1]["holds"]
    rsp, rp2 = cmp_("ddr", "pairs_synth_patch", "imagenet"), cmp_("ddr", "pairs_synth", "imagenet")
    add("S2c", 2, "SP's DDR gain over B0 is larger than P2's (point estimates)",
        holds=None if not (rsp and rp2) else bool(rsp["gain"] > rp2["gain"]),
        value="not run" if not (rsp and rp2) else f"SP {rsp['gain']:+.2f}, P2 {rp2['gain']:+.2f}")
    s2a_ran = verdict_row(rows, "S2a")["p"] is not None
    if "winner_test1" in gate and s2a_ran and sp_loc is not None:   # "not learnt" counts as not holding
        gate.update(S2a=h2["S2a"], S2b=s2b, objective="pairs_patch" if (h2["S2a"] is True or s2b is True) else "pairs")

    # ---- test 3: removing the source
    s3a = None
    if all("decode_cls_seeds" in D.columns and CODE[f] in D.index and isinstance(D.loc[CODE[f], "decode_cls_seeds"], list)
           for f in TEST3):
        drop = np.array(D.loc["MIX", "decode_cls_seeds"]) - np.array(D.loc["ADV", "decode_cls_seeds"])
        m_, lo_, hi_, _ = t_interval(drop)
        s3a = bool(m_ >= DECODE_DROP and lo_ > 0)
        add("S3a", 3, "source decodability at the CLS token at least 15 points lower for ADV than for MIX", holds=s3a,
            value=f"MIX {D.loc['MIX', 'decode_cls']:.1f}, ADV {D.loc['ADV', 'decode_cls']:.1f}; drop {m_:+.1f} "
                  f"[{lo_:+.1f}, {hi_:+.1f}]")
    else:
        add("S3a", 3, "source decodability at the CLS token at least 15 points lower for ADV than for MIX")
    note3 = "" if s3a is not False else "the manipulation check failed: not read as evidence about removing the source"
    primaries(3, [("S3b", "ADV beats MIX on ISIC", ("mix3_adv", "mix3"), 0.0)])
    rows[-1]["note"] = "; ".join(x for x in (rows[-1]["note"], note3) if x)
    any_of("S3c", 3, "ADV beats MIX on DDR or HyperKvasir (Holm over the two)",
           [(t, "mix3_adv", "mix3") for t in ("ddr", "hyperkvasir6")])
    rows[-1]["note"] = note3
    for a in TEST3:
        if r := cmp_("isic_test", a, BASE):
            extra.append(dict(comparison=f"{CODE[a]} - B1", test=LABEL["isic_test"], **r))

    # ---- exploratory: SP with the attn readout (ISIC)
    attn = [OUT / "attn" / f"ext_{cond('pairs_synth_patch', s)}.npz" for s in SCORE_SEEDS]
    if not QUICK and all(f.exists() for f in attn) and sc.has("pairs_synth_patch", "isic_test"):
        yte, _, _, ks = sc.data["isic_test"]
        C = {s: {(sp, k): np.load(f)[f"best|{sp}|{k}"][:DRAWS] == yte[sp][None] for sp in SPLITS for k in ks}
             for s, f in zip(SCORE_SEEDS, attn)}
        won = np.mean([np.load(f)[f"attn|{sp}|{k}"] for f in attn for sp in SPLITS for k in ks])
        r = sc.compare("isic_test", "pairs_synth_patch", "pairs_synth_patch", C_a=C)
        extra.append(dict(comparison=f"SP-attn - SP (attn chosen in {won:.0%} of split x k cells)",
                          test=LABEL["isic_test"], **r))
    verdict = pd.DataFrame(rows)
    return dict(levels=levels, gains=pd.DataFrame(gains), extra=pd.DataFrame(extra), diagnostics=diag,
                verdict=verdict, gate=gate, center=float(np.nanmean(cp)), learning=learning_rows())


def report(R, partial):
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("levels", "gains", "extra", "diagnostics", "verdict", "learning"):
        R[name].to_csv(OUT / f"{name}.csv", index=False)
    cell = lambda r: f"{r['gain']:+.1f} [{r['lo']:+.1f}, {r['hi']:+.1f}]"
    order = [LABEL[t] for t in TESTS]
    names = [CODE[f] for f in ["imagenet", BASE] + FAMILIES]
    lv, gn, ex, learn = R["levels"], R["gains"], R["extra"], R["learning"]
    diag = R["diagnostics"].drop(columns=["decode_cls_seeds"], errors="ignore")

    def table(df, index, rows):
        if df.empty:
            return "(none)"
        return df.pivot(index=index, columns="test", values="cell").reindex(
            index=[r for r in rows if r in set(df[index])], columns=order).to_string()

    if len(learn):
        learn_txt = learn.groupby("network").agg(min_heldout=("heldout_all", "min"), mean_heldout=("heldout_all", "mean"),
                                                 final_train=("train_acc", "mean"), minutes=("minutes", "mean"))
        learn_txt = learn_txt.reindex([n for n in names if n in learn_txt.index]).round(3).to_string()
    else:
        learn_txt = "(none)"
    b0 = gn[gn.comparison.str.endswith("- B0")] if len(gn) else gn
    other = pd.concat([gn[gn.comparison.str.endswith("- B1")] if len(gn) else gn, ex])
    L = ["Leaves to Lesions -- teaching the network to point at the anomaly (tests fixed "
         "before any code)." + (" SMOKE TEST: one split, two draws; every new condition is scored with the plant "
                                "networks." if QUICK else "") + (" The first two tests only: the third has not run."
                                                                  if partial else ""),
         "Exploratory except the predictions and the gates, which were pre-registered.",
         f"DINO ViT-S/16. Few-shot: frozen features, logistic regression (C = {Q.C}), k = {S1.KS} (HyperKvasir, "
         f"PAD-UFES-20 and HIBA {OTHER_KS}), {DRAWS} draws, {len(SPLITS)} grouped splits, balanced accuracy, mean over "
         f"k. Intervals: 95%, the wider of the hierarchical bootstrap (ISIC resamples {E10.isic_units()[1]}) and a "
         "t-interval over seed means.", "",
         "1. Learnt check (held-out pair accuracy; needs >= 0.75 for every seed):\n", learn_txt, "",
         "2. Balanced accuracy (%), mean over seeds:\n",
         table(lv.assign(cell=lv.accuracy.round(1)) if len(lv) else lv, "network", names), "",
         "3. Gains over ImageNet only (B0), points:\n",
         table(b0.assign(cell=[cell(r) for _, r in b0.iterrows()]) if len(b0) else b0, "comparison",
               [f"{n} - B0" for n in names]), "",
         "4. Against B1 on ISIC, and other comparisons (exploratory unless a prediction):\n",
         (other.assign(cell=[cell(r) for _, r in other.iterrows()])[["comparison", "test", "cell", "p", "cells"]]
          .to_string(index=False) if len(other) else "(none)"), "",
         "5. Diagnostics (mean over seeds): linear-map retention on ISIC (the stand-in's gain over B0, and its share of "
         "the network's); source decodability (five-fold probe balanced accuracy, %, chance 33); Task 1 localization "
         f"(pixel AUROC on the score half; v1 = robust PCA; the center prior scores {R['center']:.3f}):\n",
         diag.round(3).to_string(index=False), "",
         "6. Gates: " + (json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in R["gate"].items()})
                         if R["gate"] else "(not reached)"), "",
         "7. Predictions (pre-registered; primaries Holm-corrected within their test):\n",
         R["verdict"].drop(columns=["p"]).to_string(index=False)]
    text = "\n".join(L)
    (OUT / "summary.txt").write_text(text)
    print(text, flush=True)


def gate():
    t0 = time.time()
    R = results()
    g = R["gate"]
    if "objective" not in g:
        raise SystemExit(f"the gate needs the first two tests scored; it has {g}")
    json.dump(g, open(OUT / "gate.json", "w"), indent=1)
    report(R, partial=True)
    manifest("gate", gate=g, seconds=round(time.time() - t0))


def summarize():
    t0 = time.time()
    R = results()
    if (OUT / "gate.json").exists():
        old = json.load(open(OUT / "gate.json"))
        assert old.get("objective") == R["gate"].get("objective"), (old, R["gate"])
    report(R, partial=False)
    manifest("summarize", gate=R["gate"], seconds=round(time.time() - t0))


# ---------------------------------------------------------------- self-test
def selftest():
    """Test the pieces on synthetic data: quotas, the blend, lesions, the samplers, the losses, gradient reversal, the
    statistics and the task lists."""
    from PIL import Image
    from scipy import ndimage
    assert len(RUNS) == 42 and len(NETS) == 49 and NETS[7] == "pairs_species_s0" and NETS[37] == "mix3_s0"
    assert [RUNS[i] for i in (0, 24, 30, 41)] == [("pairs_species", 0), ("pairs_synth_patch", 0), ("mix3", 0),
                                                 ("mix3_adv", 5)]
    assert all(family(NETS[i + 7]) == RUNS[i][0] for i in range(42))
    r = random.Random(0)
    assert quota({"a": 10, "b": 3, "c": 100}, 30, r) == {"a": 10, "b": 3, "c": 17}
    c = quota({"x": 50, "y": 50, "z": 50}, 10, r)
    assert sum(c.values()) == 10 and max(c.values()) - min(c.values()) == 1, c
    # the blend leaves the outside unchanged and undoes a constant shift of the fill,
    # which is why the colour shift comes after the blend
    rng = np.random.RandomState(0)
    dst, src = rng.rand(40, 40, 3), rng.rand(40, 40, 3)
    mask = np.zeros((40, 40), bool)
    mask[10:30, 12:28] = True
    out = poisson(dst, src, mask)
    assert np.array_equal(out[~mask], dst[~mask]) and not np.allclose(out[mask], dst[mask])
    smooth = ndimage.gaussian_filter(rng.rand(40, 40, 3), (3, 3, 0)) * 0.5 + 0.25
    assert np.allclose(poisson(smooth, smooth + 0.1, mask), smooth, atol=1e-9)
    # a lesion stays inside the object and changes nothing outside its mask
    a = np.zeros((128, 128, 3))
    a[24:104, 20:108] = 0.2 + 0.6 * rng.rand(80, 88, 3)
    for i in range(5):
        out, m, rec = lesion(a, a[::-1, ::-1].copy(), np.random.RandomState(i))
        obj = object_mask(a)
        assert m.sum() >= 16 and not (m & ~obj).any() and np.array_equal(out[~m], a[~m]), rec
        assert rec["lesion_px"] <= 1.6 * LESION_AREA[1] * obj.sum(), rec
    # the losses and gradient reversal
    sa, sb = torch.randn(7), torch.randn(7)
    assert torch.allclose(F.softplus(sa - sb), -F.logsigmoid(sb - sa), atol=1e-6)
    x = torch.randn(3, 4, requires_grad=True)
    Reverse.apply(x, 0.5).sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, -0.5))
    x.grad = None
    Reverse.apply(x, 0.0).sum().backward()
    assert torch.count_nonzero(x.grad) == 0
    assert dann(0) == 0 and 0.999 < dann(1) < 1
    # unpaired batches: no image with its lesioned copy, every image once an epoch
    sb_ = SplitBatches(40, 8, 3)
    for _ in range(2):
        batches = list(sb_)
        assert len(batches) == len(sb_) == 10 and sorted(sum(batches, [])) == list(range(80))
        assert all(len({i % 40 for i in b_}) == len(b_) for b_ in batches)
    # patch targets: a mask filling one 16-px patch, uncropped
    m = np.zeros((224, 224), np.uint8)
    m[32:48, 64:80] = 255
    t = patch_targets(Image.fromarray(m), (0, 0, 224, 224), False)
    assert t.sum() == 1 and t[2 * GRID + 4] == 1
    t = patch_targets(Image.fromarray(m), (0, 0, 224, 224), True)
    assert t.sum() == 1 and t[2 * GRID + (GRID - 1 - 4)] == 1
    # statistics
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert boot_p(np.array([1.0] * 999 + [-1.0])) == 0.002
    mean, lo, hi, p = t_interval([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert abs(mean - 3.5) < 1e-12 and abs(hi - lo - 2 * 2.5706 * np.std([1, 2, 3, 4, 5, 6], ddof=1) / 6 ** 0.5) < 1e-3
    print("selftest passed", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["selftest", "make", "check", "train", "features", "evaluate", "diagnose", "gate",
                                     "summarize"])
    ap.add_argument("--task", type=int)
    a = ap.parse_args()
    if a.step in ("make", "train", "features", "evaluate", "diagnose"):
        {"make": make, "train": train, "features": features, "evaluate": evaluate, "diagnose": diagnose}[a.step](a.task)
    else:
        {"selftest": selftest, "check": check, "gate": gate, "summarize": summarize}[a.step]()
