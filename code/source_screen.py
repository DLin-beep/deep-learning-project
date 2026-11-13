"""Which source domains help which target domains, and does sharing low-level visual attributes predict it?

This is the screen. Later experiments import it (as S1) and train their sources with train(), so those
sources are defined here as well.

Sources (intermediate training, size-matched to DTD, 3 seeds each):
  plant  PlantVillage, segmented leaves (38 classes)
  food   Food-101, 38 random classes (control: fine-grained, no disease)
  histo  NCT-CRC-HE-100K colorectal histopathology (9 classes)
  dtd    Describable Textures (45 classes; 'freckled' and 'veined' removed because they are mostly skin and leaves)
Targets (few-shot, frozen features):
  ham    HAM10000 dermoscopy (7 classes, split by lesion)
  ddr    DDR fundus photographs, diabetic-retinopathy grades 0-4 (split by image, since no patient ids are released)
  cxr    ChestX-ray14, the 7-class single-label subset of Guo et al. (split by patient, capped at ~12k images)
Backbones (all ImageNet-1k): rn50_sup (supervised ResNet-50), rn50_dino (DINO ResNet-50),
  vits_sup (supervised ViT-S/16, DeiT-S), vits_dino (DINO ViT-S/16).

For every backbone x target, the ImageNet-only network, every source run and each run's normalisation-only control
are evaluated as in our corrected pilot. The control keeps the ImageNet weights but takes the source
run's BatchNorm running statistics of layer3/4 (ResNets) or LayerNorm affine of blocks 7-12 (ViTs). The evaluation
uses 5 grouped splits, k = 5/20/50 shots x 20 draws, the readout chosen on validation and a hierarchical bootstrap.

Attribute distance is the Frechet distance between hand-crafted low-level feature distributions (dots and blobs at
several scales, edges, ridges, orientation, local texture, colour), with and without colour. Overall similarity is
the Frechet distance between supervised ResNet-50 ImageNet features (Matsoukas et al.).

Steps, each run as its own SLURM job:
  train --backbone B --source S --seed T    source training (GPU)
  cache --target X                          uint8 evaluation images + index (CPU)
  extract --backbone B --target X           features of every condition trained on that backbone (GPU)
  evaluate --backbone B --target X          few-shot predictions of the same conditions (CPU)
  fulldata --backbone B --target X          full-data ceiling of every non-control condition (CPU)
  attributes                                attribute and ImageNet-feature distances (GPU)
  summarize                                 tables, transfer gains and the cross-pair analysis (CPU)
  tasks                                     prints the train task indices of the PlantVillage ablations
"""
import argparse
import io
import itertools
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFilter

for _d in sorted(Path(__file__).resolve().parents[1].glob("[1-9]_*")):   # the paper's nine experiments
    sys.path.insert(0, str(_d))
import e1_pilot as P   # noqa: E402
import e1_pilot2 as Q  # noqa: E402
from common import backbones as _B, bootstrap as _BS, data as _D, features as _F, fewshot as _FS, heads as _H  # noqa: E402
from common import splits as _SP  # noqa: E402
from common.backbones import (CLIP_MEAN, CLIP_STD, MICRO, RES, ctrl, family, head_keys, hf_clip_vision,  # noqa: E402
                              imagenet_model, is_vit, last, micro_batch, n_norm_params, norm_param, norm_stats,
                              panderm_vision, readouts, res, stage_features, stages, transforms_at)
from common.data import CXR_CLASSES, CXR_MAX, TARGETS, cpus, n_classes, target_table  # noqa: E402
from common.fewshot import eval_job as _eval_job  # noqa: E402
from common.heads import readout  # noqa: E402

OUT = P.ROOT / "pilot" / "out_s1"
DATA = P.DATA
MAIN_SOURCES = ["plant", "food", "histo", "dtd"]
ABLATIONS = ["plant_gray", "plant_blur", "plant_shuf"]   # PlantVillage without colour / fine dots / leaf shape
EDGE_CONTROLS = ["plant_grid", "plant_gridshuf"]         # tile borders with / without shuffling (DINO ViT-S only)
OBJECT_CONTROLS = ["coil"]                               # COIL-100: isolated objects on black, no plants (DINO ViT-S)
LABEL_CONTROLS = ["plant_species", "plant_disease2"]     # PlantVillage images, species / healthy-vs-diseased labels
# The sources below belong to later experiments. They are DINO ViT-S only and not in SOURCES, so the steps of earlier
# experiments skip them.
# The abnormality-sources test uses abnormality vs identity labels on the same
# images of two non-leaf datasets, VisA (12 inspected object types) and FruitNet (6 fruits).
ABNORMALITY_SOURCES = ["visa_defect", "visa_object", "fruit_quality", "fruit_type"]
CROP_SOURCES = ["fruit_quality_crop", "fruit_type_crop"]  # FruitNet cropped to the fruit
ABNORMAL_LABELS = ["visa_defect", "fruit_quality", "fruit_quality_crop"]  # the others are identity labels
# The why-sources-transfer test asks which property of a source carries the
# transfer.
HISTO_CANCER = ["TUM", "STR", "DEB"]       # tumour epithelium, cancer-associated stroma, debris / necrosis
HISTO_ARBITRARY = ["ADI", "LYM", "MUC"]    # the same 1 : 2 balance, no tumour class
PLANT_DIV = {"plant_div2": ["Tomato", "Corn_(maize)"],
             "plant_div5": ["Tomato", "Corn_(maize)", "Apple", "Pepper,_bell", "Cherry_(including_sour)"],
             "plant_div8": ["Tomato", "Corn_(maize)", "Apple", "Pepper,_bell", "Cherry_(including_sour)", "Grape",
                            "Strawberry", "Peach"]}
N_DIV = 3600                               # the largest size at which plant_div2 can be balanced (Corn: 1,162 healthy)
STAGE12_SOURCES = ["histo_cancer", "histo_arbitrary2"] + list(PLANT_DIV) + ["pool_abnormal", "pool_identity"]
# The source-size test trains on PlantVillage healthy vs diseased at six sizes, with the
# natural class mix, nested within seed. The "_840steps" variants train for 840 steps whatever the size.
SIZE_N = [1350, 2700, 5400, 10800, 21600, 43200]
SIZE_SOURCES = [f"plant_n{n}" for n in SIZE_N] + [f"plant_n{n}_840steps" for n in SIZE_N if n != 5400]
# The lesion-like-sources test uses sources with lesion-like marks but no plants.
# MatsyaDx-BD fish photos, cropped to the fish, are labelled skin disease vs healthy or by species on the same images.
# COIL-100 gets painted blotches.
STAGE18_SOURCES = ["fish_disease", "fish_species", "coil_blotch"]
N_FISH = 1350                              # the smallest size with the full gain in the source-size test
# The second-leaf-dataset test uses a second leaf-disease dataset
# (Chouhan et al. 2019), labelled diseased vs healthy or by species on the same images.
STAGE19_SOURCES = ["leaf_disease", "leaf_species"]
N_LEAF = 1350
# The leaf-crop test uses leaf_disease's images per seed, cropped so the leaf fills the
# frame.
STAGE20_SOURCES = ["leaf_disease_crop"]
# The defect-crop test, on framing beyond plants, uses visa_defect's images per seed,
# cropped so the objects fill the frame.
STAGE21_SOURCES = ["visa_defect_crop"]
# A lighter plant training for the unlabeled-skin test, meant to keep more of
# ImageNet's features. It uses plant_disease2's images and steps at a tenth of the learning rate.
STAGE26_SOURCES = ["plant_disease2_lite"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
SOURCES = MAIN_SOURCES + ABLATIONS + EDGE_CONTROLS + OBJECT_CONTROLS + LABEL_CONTROLS
BACKBONES = ["rn50_sup", "rn50_dino", "vits_sup", "vits_dino"]
REF_BACKBONES = ["clip_b16"]                            # frozen reference only (no source training)
ALL_BACKBONES = BACKBONES + REF_BACKBONES
N_PERM = 20000                                          # Monte Carlo permutations when exact enumeration is too big
SEEDS = [0, 1, 2]
SPLITS = [0, 1, 2, 3, 4]
KS = [5, 20, 50]
REPEATS = 20
DTD_EXCLUDE = ["freckled", "veined"]       # mostly human skin / leaves (checked on contact sheets)
N_SOURCE = 45 * 120                         # DTD's size after the exclusion. Every source is sampled to this size.
N_ATTR = 1500                               # images per dataset for the hand-crafted attribute features
N_INET = 3000                               # images per dataset for the ImageNet-feature Frechet distance
EPOCHS, BATCH, LR = P.EPOCHS, P.BATCH, P.LR


# ---------------------------------------------------------------- backbones
def conditions(b, sources=None):
    if b in REF_BACKBONES:
        return ["imagenet"]
    runs = [f"{s}_s{t}" for s in (sources or SOURCES) for t in SEEDS]
    return ["imagenet"] + runs + [ctrl(b) + r for r in runs]


def trained_conditions(b):
    """conditions(b) for the sources trained on this backbone, with a checkpoint for every seed. Some sources were
    trained on DINO ViT-S only, so the extract and evaluate steps skip them on the other backbones."""
    if b in REF_BACKBONES:
        return conditions(b)
    srcs = [s for s in SOURCES if all((OUT / b / "ckpt" / f"{s}_s{t}.pt").exists() for t in SEEDS)]
    if len(srcs) < len(SOURCES):
        print(f"{b}: skipping sources without a checkpoint for every seed: "
              f"{', '.join(s for s in SOURCES if s not in srcs)}", flush=True)
    return conditions(b, srcs) if srcs else ["imagenet"]


# ---------------------------------------------------------------- sources
def _class_dirs(root, exclude=()):
    return sorted(d.name for d in Path(root).iterdir() if d.is_dir() and d.name not in exclude)


def _quota_sample(per_class_files, rng, n_total):
    """Size-match: n_total images, as equal per class as possible (the remainder goes to random classes)."""
    n_c = len(per_class_files)
    quota, extra = divmod(n_total, n_c)
    plus = set(rng.sample(range(n_c), extra))
    items = []
    for y, files in enumerate(per_class_files):
        items += [(f, y) for f in rng.sample(files, quota + (y in plus))]
    return items


class Gray:
    """Remove colour."""
    def __call__(self, img):
        return img.convert("L").convert("RGB")


class Blur:
    """Remove fine dots and specks (Gaussian blur, radius 3 px on the 256 px PlantVillage images)."""
    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=3))


class TileShuffle:
    """Remove the global leaf shape by cutting the image into a 4x4 grid and permuting the tiles at random (Neyshabur
    et al.). Local texture and colour are kept."""
    def __init__(self, n=4):
        self.n = n

    def __call__(self, img):
        w, h = img.size
        tw, th = w // self.n, h // self.n
        tiles = [img.crop((j * tw, i * th, (j + 1) * tw, (i + 1) * th)) for i in range(self.n) for j in range(self.n)]
        random.shuffle(tiles)
        out = Image.new(img.mode, (tw * self.n, th * self.n))
        for k, tile in enumerate(tiles):
            out.paste(tile, ((k % self.n) * tw, (k // self.n) * th))
        return out


class TileGrid(TileShuffle):
    """Control for TileShuffle's tile edges. Draws black 4 px borders between the 4x4 tiles of every image, with the
    tiles left in place (plant_grid, leaf layout kept) or permuted (plant_gridshuf, layout removed). Both have the
    same edges, and the borders hide the seams that shuffling creates."""
    def __init__(self, shuffle, n=4, width=4):
        super().__init__(n)
        self.shuffle, self.width = shuffle, width

    def __call__(self, img):
        from PIL import ImageDraw
        out = super().__call__(img) if self.shuffle else img.copy()
        w, h = out.size
        draw = ImageDraw.Draw(out)
        for i in range(1, self.n):
            x, y = i * (w // self.n), i * (h // self.n)
            draw.rectangle([x - self.width // 2, 0, x + self.width // 2 - 1, h - 1], fill=(0, 0, 0))
            draw.rectangle([0, y - self.width // 2, w - 1, y + self.width // 2 - 1], fill=(0, 0, 0))
        return out


ABLATION_OPS = {"plant_gray": Gray(), "plant_blur": Blur(), "plant_shuf": TileShuffle(),
                "plant_grid": TileGrid(shuffle=False), "plant_gridshuf": TileGrid(shuffle=True)}


def _abnormality_table(source):
    """The index written when the abnormality sources were prepared (256 px images). Each image has its group (object type
    or fruit), whether it is abnormal (defective / bad quality) and whether it is in the fixed held-out set."""
    if source.startswith("visa"):
        root = DATA / "visa256"
        df = pd.read_csv(root / "index.csv")
        return df.assign(path=[str(root / p) for p in df.path], group=df.object, abnormal=df.label == 1)
    root = DATA / ("fruit256crop" if source.endswith("_crop") else "fruit256")
    df = pd.read_csv(root / "index.csv")
    return df.assign(path=[str(root / p) for p in df.path], group=df.fruit, abnormal=df.quality == "bad")


def _abnormality_items(source, rng, n_total):
    """[(path, (group, abnormal))] and the sorted groups. Both label sets of a dataset draw the same images per seed.
    VisA takes every non-held-out defective image (as equal per object type as possible, at most 2/9 of the sample)
    and fills the rest with normal ones. FruitNet takes equal numbers of good and bad images of every fruit."""
    df = _abnormality_table(source)
    df = df[~df.heldout]
    groups = sorted(df.group.unique())
    cells = [(g, ab) for ab in (True, False) for g in groups]
    files = [sorted(df[(df.group == g) & (df.abnormal == ab)].path) for g, ab in cells]
    if source.startswith("visa"):
        n = len(groups)
        n_def = min(sum(len(f) for f in files[:n]), round(n_total * 2 / 9))
        items = _quota_sample(files[:n], rng, n_def) + \
            [(f, c + n) for f, c in _quota_sample(files[n:], rng, n_total - n_def)]
    else:
        items = _quota_sample(files, rng, n_total)
    return [(f, cells[c]) for f, c in items], groups


def _plant_div_items(source, rng, n_total):
    """PlantVillage healthy vs diseased, over the species in PLANT_DIV[source]. Each species gets the same number of
    images, half healthy and half diseased, so the species never gives the label away. The diseased half is split
    equally across that species' diseases."""
    root = DATA / "plantvillage" / "repo" / "raw" / "segmented"
    classes, per = _class_dirs(root), n_total // len(PLANT_DIV[source])
    files = lambda c: sorted(str(p) for p in (root / c).iterdir() if p.suffix.lower() in IMG_EXTS)
    items = []
    for s in PLANT_DIV[source]:
        mine = [c for c in classes if c.split("___")[0] == s]
        healthy, sick = [c for c in mine if "healthy" in c], [c for c in mine if "healthy" not in c]
        items += [(f, 0) for f in rng.sample(files(healthy[0]), per // 2)]
        items += [(f, 1) for f, _ in _quota_sample([files(c) for c in sick], rng, per - per // 2)]
    return items, ["healthy", "diseased"]


def _pool_identities():
    """The 32 object identities of the pooled source (14 plant species, 12 VisA objects, 6 fruits), in a fixed order."""
    plant = sorted({c.split("___")[0] for c in _class_dirs(DATA / "plantvillage" / "repo" / "raw" / "segmented")})
    visa = sorted(_abnormality_table("visa_defect").group.unique())
    fruit = sorted(_abnormality_table("fruit_quality_crop").group.unique())
    return [f"plant:{x}" for x in plant] + [f"visa:{x}" for x in visa] + [f"fruit:{x}" for x in fruit]


def _pool_items(rng, n_total):
    """[(path, abnormal, identity)] from equal thirds of PlantVillage (38-class sampling), VisA, and FruitNet cropped
    to the fruit. Both pooled label sets draw the same images per seed."""
    third = n_total // 3
    plant, classes = source_items("plant", rng, third)
    out = [(f, int("healthy" not in classes[y]), "plant:" + classes[y].split("___")[0]) for f, y in plant]
    for src, tag, n in [("visa_defect", "visa", third), ("fruit_quality_crop", "fruit", n_total - 2 * third)]:
        out += [(f, int(ab), f"{tag}:{g}") for f, (g, ab) in _abnormality_items(src, rng, n)[0]]
    return out


def _plantvillage_binary():
    """Every PlantVillage image as (path, 0 healthy / 1 diseased), for the source-size test. Returns the
    pool that training sets are drawn from and a fixed held-out set of 1,000 images that no size ever trains on."""
    root = DATA / "plantvillage" / "repo" / "raw" / "segmented"
    every = [(str(p), int("healthy" not in c)) for c in _class_dirs(root)
             for p in sorted((root / c).iterdir()) if p.suffix.lower() in IMG_EXTS]
    held = set(random.Random(7).sample(range(len(every)), 1000))
    return [x for i, x in enumerate(every) if i not in held], [x for i, x in enumerate(every) if i in held]


def _fish_table():
    """MatsyaDx-BD skin-disease and healthy photos cropped to the fish, for the lesion-like-sources test. Includes
    condition, species, specimen and the fixed held-out fish."""
    root = DATA / "fish256crop"
    df = pd.read_csv(root / "index.csv")
    return df.assign(path=[str(root / p) for p in df.path], diseased=(df.condition != "Healthy Fish").astype(int))


def _leaf_table():
    """Chouhan et al. leaf photos letterboxed to 256 px, for the second-leaf-dataset test. Includes species, condition
    and the fixed held-out set."""
    root = DATA / "leaf256"
    df = pd.read_csv(root / "index.csv")
    return df.assign(path=[str(root / p) for p in df.path], diseased=(df.condition == "diseased").astype(int))


def _leaf_crop(items):
    """The same leaf images, from the copies cropped to the leaf (the leaf-crop test)."""
    return [(str(DATA / "leaf256crop" / Path(f).name), y) for f, y in items]


def _visa_crop(items):
    """The same VisA images, from the copies cropped to their objects (the defect-crop test)."""
    return [(str(DATA / "visa256crop" / Path(f).relative_to(DATA / "visa256")), y) for f, y in items]


def _coil_blotch(items, rng):
    """Half of each object's sampled COIL-100 views (drawn with rng) swapped for their blotched copies, for the
    lesion-like-sources test. Labels are 1 blotched, 0 clean."""
    per = {}
    for f, y in items:
        per.setdefault(y, []).append(f)
    out = []
    for y in sorted(per):
        hit = set(rng.sample(per[y], len(per[y]) // 2))
        out += [(str(DATA / "coil100_blotch" / Path(f).name), 1) if f in hit else (f, 0) for f in per[y]]
    return out


def size_n(source):
    return int(source.split("_")[1][1:])           # "plant_n43200_840steps" -> 43200


def fixed_steps(source):
    """Training steps for the fixed-step arm of the source-size test. None means the screen's 10 epochs."""
    return 840 if source.endswith("_840steps") else None


def heldout_items(source, items=None, seed=0):
    """Held-out images of a source task, labelled as in training, to check the task was learnt ([] if it has none).

    The why-sources-transfer sources and histo's added seeds use up to 1,000 images of a second draw from
    the same source (seed 100000 + seed) that this network did not train on. The source-size sources use
    the fixed held-out PlantVillage set, and the fish the 6 held-out fish. For the blotches, it is 1,000
    of the views this seed did not train on, half of them blotched. The abnormality, leaf and crop sources use their
    dataset's fixed held-out set."""
    if source in SIZE_SOURCES:
        return _plantvillage_binary()[1]
    if source in ("fish_disease", "fish_species"):
        h = _fish_table()
        h = h[h.heldout]
        if source == "fish_disease":
            return [(p, int(d)) for p, d in zip(h.path, h.diseased)]
        species = sorted(_fish_table().species.unique())
        return [(p, species.index(s)) for p, s in zip(h.path, h.species)]
    if source == "leaf_disease_crop":
        return _leaf_crop(heldout_items("leaf_disease"))
    if source == "visa_defect_crop":
        return _visa_crop(heldout_items("visa_defect"))
    if source in ("leaf_disease", "leaf_species"):
        t = _leaf_table()
        species, h = sorted(t.species.unique()), t[t.heldout]
        if source == "leaf_disease":
            return [(p, int(d)) for p, d in zip(h.path, h.diseased)]
        return [(p, species.index(s)) for p, s in zip(h.path, h.species)]
    if source == "coil_blotch" and items is not None:
        used = {Path(f).name for f, _ in items}
        rest = [p for p in sorted((DATA / "coil100" / "coil-100").glob("obj*__*.png")) if p.name not in used]
        r = random.Random(7)
        pick = r.sample(rest, min(1000, len(rest)))
        hit = set(r.sample(range(len(pick)), len(pick) // 2))
        return [(str(DATA / "coil100_blotch" / p.name), 1) if i in hit else (str(p), 0) for i, p in enumerate(pick)]
    if source in STAGE12_SOURCES + ["histo"] and items is not None:
        used = {f for f, _ in items}
        extra = [(f, y) for f, y in source_items(source, random.Random(100000 + seed))[0] if f not in used]
        return random.Random(7).sample(extra, min(1000, len(extra)))
    if source not in ABNORMALITY_SOURCES + CROP_SOURCES:
        return []
    df = _abnormality_table(source)
    groups = sorted(df[~df.heldout].group.unique())
    h = df[df.heldout]
    if source in ABNORMAL_LABELS:
        return [(p, int(a)) for p, a in zip(h.path, h.abnormal)]
    return [(p, groups.index(g)) for p, g in zip(h.path, h.group)]


def source_items(source, rng, n_total=None):
    n_total = n_total or N_SOURCE
    if source in ABLATION_OPS:                     # the same sampled PlantVillage images, with the op applied on load
        return source_items("plant", rng, n_total)
    if source in ABNORMALITY_SOURCES + CROP_SOURCES:   # abnormal vs normal, or object type / fruit, on the same images
        items, groups = _abnormality_items(source, rng, n_total)
        if source in ABNORMAL_LABELS:
            return [(f, int(ab)) for f, (g, ab) in items], ["normal", "abnormal"]
        return [(f, groups.index(g)) for f, (g, ab) in items], groups
    if source == "plant_species":                  # the same images, labelled by species only (no disease to learn)
        items, classes = source_items("plant", rng, n_total)
        species = sorted({c.split("___")[0] for c in classes})
        to_species = [species.index(c.split("___")[0]) for c in classes]
        return [(f, to_species[y]) for f, y in items], species
    if source == "plant_disease2_lite":            # the same images, trained at a tenth of the learning rate
        return source_items("plant_disease2", rng, n_total)
    if source == "plant_disease2":                 # the same images, labelled healthy vs diseased only
        items, classes = source_items("plant", rng, n_total)
        return [(f, int("healthy" not in classes[y])) for f, y in items], ["healthy", "diseased"]
    if source in ("histo_cancer", "histo_arbitrary2"):   # the same tissue tiles, two binary label sets
        items, classes = source_items("histo", rng, n_total)
        group = HISTO_CANCER if source == "histo_cancer" else HISTO_ARBITRARY
        return [(f, int(classes[y] in group)) for f, y in items], ["other", "+".join(group)]
    if source in PLANT_DIV:                        # diversity of abnormalities, species-balanced
        return _plant_div_items(source, rng, min(n_total, N_DIV))
    if source in ("pool_abnormal", "pool_identity"):   # PlantVillage + VisA + cropped FruitNet
        items = _pool_items(rng, n_total)
        if source == "pool_abnormal":
            return [(f, ab) for f, ab, _ in items], ["normal", "abnormal"]
        names = _pool_identities()
        return [(f, names.index(i)) for f, _, i in items], names
    if source in SIZE_SOURCES:                     # the first n images of one random order per seed
        pool, _ = _plantvillage_binary()
        return rng.sample(pool, len(pool))[:size_n(source)], ["healthy", "diseased"]
    if source in ("fish_disease", "fish_species"):   # the same fish photos per seed, two label sets
        pool = _fish_table()
        species = sorted(pool.species.unique())
        pool = pool[~pool.heldout].reset_index(drop=True)
        rows = rng.sample(range(len(pool)), min(N_FISH, len(pool)))
        if source == "fish_disease":
            return [(pool.path[i], int(pool.diseased[i])) for i in rows], ["healthy", "skin disease"]
        return [(pool.path[i], species.index(pool.species[i])) for i in rows], species
    if source == "leaf_disease_crop":              # leaf_disease's images, cropped to the leaf
        items, classes = source_items("leaf_disease", rng, n_total)
        return _leaf_crop(items), classes
    if source == "visa_defect_crop":               # visa_defect's images, cropped to the objects
        items, classes = source_items("visa_defect", rng, n_total)
        return _visa_crop(items), classes
    if source in ("leaf_disease", "leaf_species"):   # the same leaf photos per seed, two label sets
        pool = _leaf_table()
        species = sorted(pool.species.unique())
        pool = pool[~pool.heldout].reset_index(drop=True)
        rows = rng.sample(range(len(pool)), min(N_LEAF, len(pool)))
        if source == "leaf_disease":
            return [(pool.path[i], int(pool.diseased[i])) for i in rows], ["healthy", "diseased"]
        return [(pool.path[i], species.index(pool.species[i])) for i in rows], species
    if source == "coil_blotch":                    # the COIL-100 control's images, half of them blotched
        items, _ = source_items("coil", rng, n_total)
        return _coil_blotch(items, rng), ["clean", "blotched"]
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    if source in ("plant", "histo", "dtd"):
        root = {"plant": DATA / "plantvillage" / "repo" / "raw" / "segmented",
                "histo": DATA / "nct" / "NCT-CRC-HE-100K",
                "dtd": DATA / "dtd" / "dtd" / "images"}[source]
        classes = _class_dirs(root, DTD_EXCLUDE if source == "dtd" else ())
        files = [sorted(str(p) for p in (root / c).iterdir() if p.suffix.lower() in exts) for c in classes]
        return _quota_sample(files, rng, n_total), classes
    if source == "food":
        df = pd.concat([pd.read_parquet(p) for p in sorted((DATA / "food101").glob("*.parquet"))])
        quota = -(-n_total // 38)
        counts = df["label"].value_counts()
        chosen = sorted(rng.sample(sorted(counts[counts >= quota].index.tolist()), 38))
        files = [[r["bytes"] for r in df[df["label"] == lab]["image"]] for lab in chosen]
        return _quota_sample(files, rng, n_total), [int(c) for c in chosen]
    if source == "coil":                            # 100 objects x 72 views, 128 px, black background
        per = {}
        for p in sorted((DATA / "coil100" / "coil-100").glob("obj*__*.png")):
            per.setdefault(int(p.name.split("__")[0][3:]), []).append(str(p))
        return _quota_sample([per[i] for i in sorted(per)], rng, n_total), [f"obj{i}" for i in sorted(per)]
    raise ValueError(source)


def train(b, source, seed, steps=None, ckpt_dir=None):
    """Train one source network with the pilots' recipe, at the backbone's resolution.

    A ResNet trains layer3/4 and the head (SGD, wd 1e-4). A ViT trains the second half of its blocks (7-12 of 12), the
    final norm and the head (SGD, no weight decay, gradient clipping at 1). steps and ckpt_dir override the step count
    and the checkpoint folder, for smoke tests."""
    ck = (ckpt_dir or OUT / b / "ckpt") / f"{source}_s{seed}.pt"
    if ck.exists():
        print("exists:", ck); return
    P.seed_all(seed)
    rng = random.Random(seed)
    items, classes = source_items(source, rng)
    print(f"{b} {source} seed {seed}: {len(items)} images, {len(classes)} classes", flush=True)
    from torchvision import transforms
    train_tf, eval_tf = transforms_at(b)
    tf = (transforms.Compose([ABLATION_OPS[source], *train_tf.transforms]) if source in ABLATION_OPS
          else train_tf)
    if norm_stats(b) != (P.MEAN, P.STD):           # CLIP networks: their own normalisation, as when features are read
        tf = transforms.Compose([*tf.transforms[:-1], transforms.Normalize(*norm_stats(b))])
    dl = torch.utils.data.DataLoader(P.ImageList(items, tf), batch_size=BATCH, shuffle=True,
                                     num_workers=min(6, cpus()), drop_last=True, persistent_workers=True)
    dev = P.device()
    m = imagenet_model(b)
    if is_vit(b):
        m.reset_classifier(len(classes))
        half = range(12, 24) if b.startswith("vitl") else Q.VIT_TRAINED    # the second half of the blocks
        trainable = tuple(f"blocks.{i}." for i in half) + ("norm.", "head.")
        for n, p in m.named_parameters():
            p.requires_grad = n.startswith(trainable)
        frozen_mods = []
        wd, clip = 0.0, 1.0
    else:
        m.fc = nn.Linear(2048, len(classes))
        frozen_mods = ["conv1", "bn1", "layer1", "layer2"]
        for name in frozen_mods:
            for p in getattr(m, name).parameters():
                p.requires_grad = False
        wd, clip = P.WD, None
    m.to(dev)
    params = [p for p in m.parameters() if p.requires_grad]
    lr = LR / 10 if source in STAGE26_SOURCES else LR
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    steps = steps or fixed_steps(source) or EPOCHS * len(dl)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    lossf = nn.CrossEntropyLoss()
    micro = micro_batch(b)                         # ViT-L at 224 px: 8 micro-batches of 8, so it fits an 8 GB GPU
    log, done = [], 0
    for ep in range(-(-steps // len(dl))):        # EPOCHS, but the fixed-step arm may end mid-epoch
        m.train()
        for name in frozen_mods:
            getattr(m, name).eval()
        t0, tot, correct, n = time.time(), 0.0, 0, 0
        for x, yb in dl:
            if done == steps:
                break
            done += 1
            x, yb = x.to(dev), yb.to(dev)
            if micro:                              # large networks: the same update, summed over micro-batches
                opt.zero_grad()
                parts = []
                for xs, ys in zip(x.split(micro), yb.split(micro)):
                    out = m(xs)
                    (lossf(out, ys) * len(ys) / len(yb)).backward()
                    parts.append(out.detach())
                logits = torch.cat(parts)
                loss = lossf(logits, yb)
            else:
                logits = m(x)
                loss = lossf(logits, yb)
                opt.zero_grad(); loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step(); sched.step()
            tot += loss.item() * len(yb); correct += (logits.argmax(1) == yb).sum().item(); n += len(yb)
        log.append(dict(epoch=ep + 1, loss=tot / n, train_acc=correct / n, sec=round(time.time() - t0)))
        print(log[-1], flush=True)
    heldout_ba, held = None, heldout_items(source, items, seed)
    if held:                                       # was the source task learnt? (the head is still attached)
        m.eval()
        pred = []
        with torch.no_grad():
            for x, _ in torch.utils.data.DataLoader(P.ImageList(held, eval_tf), batch_size=128,
                                                    num_workers=min(6, cpus())):
                pred.append(m(x.to(dev)).argmax(1).cpu())
        pred, yh = torch.cat(pred).numpy(), np.array([y for _, y in held])
        heldout_ba = float(np.mean([(pred[yh == c] == c).mean() for c in np.unique(yh)]))   # classes present
        print(f"held-out balanced accuracy on the source task ({len(held)} images): {heldout_ba:.3f}", flush=True)
    ck.parent.mkdir(parents=True, exist_ok=True)
    drop = tuple(f"{k.split('.')[0]}." for k in head_keys(b))
    tmp = ck.with_suffix(".tmp")
    torch.save({k: v.cpu() for k, v in m.state_dict().items() if not k.startswith(drop)}, tmp)
    os.replace(tmp, ck)
    json.dump(dict(backbone=b, source=source, seed=seed, n=len(items), classes=len(classes), steps=steps, lr=lr, log=log,
                   heldout_ba=heldout_ba), open(ck.with_suffix(".json"), "w"), indent=1)


# ---------------------------------------------------------------- attributes
ATTR_NAMES = ([f"{kind}_dog{s}" for s in (1, 2, 4, 8) for kind in ("bright", "dark", "energy")]
              + ["sobel_mean", "sobel_std", "canny1", "canny3", "ridge_mean", "ridge_p95", "coherence"]
              + [f"lbp{i}" for i in range(10)]
              + ["L_mean", "L_std", "a_mean", "a_std", "b_mean", "b_std", "sat_mean", "sat_std", "colourfulness"])
COLOUR = ATTR_NAMES[-9:]


def attr_vector(rgb):
    """Hand-crafted low-level attributes of one 224x224 RGB uint8 image.

    They cover dots and blobs at four scales (difference of Gaussians, with bright and dark local extrema per 10k
    pixels and the response energy), edges, ridges and streaks, orientation coherence, local texture (uniform LBP
    histogram) and colour."""
    from scipy import ndimage as ndi
    from skimage.color import rgb2gray, rgb2hsv, rgb2lab
    from skimage.feature import canny, local_binary_pattern
    from skimage.filters import frangi, sobel
    x = rgb.astype(np.float64) / 255
    g = rgb2gray(x)
    f = []
    for s in (1, 2, 4, 8):
        dog = ndi.gaussian_filter(g, s) - ndi.gaussian_filter(g, 1.6 * s)
        for sign in (1, -1):
            r = sign * dog
            peaks = (r == ndi.maximum_filter(r, size=2 * s + 1)) & (r > 0.02)
            f.append(peaks.sum() / g.size * 1e4)
        f.append(np.abs(dog).mean())
    sb = sobel(g)
    f += [sb.mean(), sb.std(), canny(g, sigma=1).mean(), canny(g, sigma=3).mean()]
    fr = frangi(g, sigmas=(1, 2, 3), black_ridges=True)
    f += [fr.mean(), np.percentile(fr, 95)]
    gx, gy = ndi.sobel(g, 1), ndi.sobel(g, 0)
    jxx, jyy, jxy = (ndi.gaussian_filter(v, 2) for v in (gx * gx, gy * gy, gx * gy))
    f.append((np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-8)).mean())
    lbp = local_binary_pattern((g * 255).astype(np.uint8), P=8, R=1, method="uniform").astype(int)
    f += list(np.bincount(lbp.ravel(), minlength=10)[:10] / lbp.size)
    lab, hsv = rgb2lab(x), rgb2hsv(x)
    for ch in range(3):
        f += [lab[..., ch].mean(), lab[..., ch].std()]
    f += [hsv[..., 1].mean(), hsv[..., 1].std()]
    rg, yb = x[..., 0] - x[..., 1], 0.5 * (x[..., 0] + x[..., 1]) - x[..., 2]
    f.append(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))
    return np.array(f, np.float64)


def frechet(a, b):
    from scipy import linalg
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(len(s1)) * 1e-6
        covmean = linalg.sqrtm((s1 + off) @ (s2 + off))
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * np.real(covmean)))


def sym_distances(names, fn):
    d = {a: {} for a in names}
    for i, a in enumerate(names):
        d[a][a] = 0.0
        for b in names[i + 1:]:
            d[a][b] = d[b][a] = fn(a, b)
    return d


def dataset_images(name, n, seed=0):
    """n 224x224 RGB uint8 images (N, 224, 224, 3), from a source's seed-0 training sample or a target's cache."""
    from torchvision import transforms
    rng = np.random.RandomState(seed)
    if name in TARGETS:
        tab, images = load_target(name)
        idx = np.sort(rng.choice(len(tab), min(n, len(tab)), replace=False))
        return np.ascontiguousarray(images[idx].transpose(0, 2, 3, 1))
    items, _ = source_items(name, random.Random(0))
    pick = rng.choice(len(items), min(n, len(items)), replace=False)
    op = ABLATION_OPS.get(name)
    random.seed(seed)
    tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224)])
    out = []
    for i in pick:
        src = items[i][0]
        img = Image.open(io.BytesIO(src) if isinstance(src, bytes) else src).convert("RGB")
        out.append(np.asarray(tf(op(img) if op else img)))
    return np.stack(out)


@torch.no_grad()
def attributes():
    from joblib import Parallel, delayed
    # The 10 datasets of the stored attributes.json. The features are standardised over every listed dataset, so adding
    # the sources that came later would change every distance.
    names = MAIN_SOURCES + ABLATIONS + TARGETS
    res = {"n_attr": N_ATTR, "n_inet": N_INET, "attr_names": ATTR_NAMES, "colour": COLOUR}
    feats = {}
    for nm in names:
        t0 = time.time()
        imgs = dataset_images(nm, N_ATTR)
        feats[nm] = np.stack(Parallel(n_jobs=cpus())(delayed(attr_vector)(im) for im in imgs))
        print(f"attributes {nm}: {len(imgs)} images in {time.time() - t0:.0f}s", flush=True)
    pooled = np.concatenate(list(feats.values()))
    mu, sd = pooled.mean(0), pooled.std(0) + 1e-12
    z = {nm: (v - mu) / sd for nm, v in feats.items()}
    nocol = [i for i, a in enumerate(ATTR_NAMES) if a not in COLOUR]
    res["attr"] = sym_distances(names, lambda a, b: frechet(z[a], z[b]))
    res["attr_nocolour"] = sym_distances(names, lambda a, b: frechet(z[a][:, nocol], z[b][:, nocol]))
    res["attr_means"] = {nm: dict(zip(ATTR_NAMES, feats[nm].mean(0).tolist())) for nm in names}
    # Overall similarity: supervised ResNet-50 ImageNet features (global-pooled layer4).
    dev = P.device()
    m = imagenet_model("rn50_sup").eval().to(dev)
    mean = torch.tensor(P.MEAN).view(1, 3, 1, 1).to(dev); std = torch.tensor(P.STD).view(1, 3, 1, 1).to(dev)
    inet = {}
    for nm in names:
        imgs = dataset_images(nm, N_INET, seed=1)
        fs = []
        for s in range(0, len(imgs), 256):
            x = torch.from_numpy(imgs[s:s + 256]).permute(0, 3, 1, 2).to(dev).float().div_(255)
            fs.append(stage_features("rn50_sup", m, (x - mean) / std)["s4"].cpu().numpy().astype(np.float64))
        inet[nm] = np.concatenate(fs)
        print(f"imagenet features {nm}: {len(imgs)} images", flush=True)
    res["inet"] = sym_distances(names, lambda a, b: frechet(inet[a], inet[b]))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "attributes.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ["attr", "attr_nocolour", "inet"]}, indent=1)[:3000])


# ---------------------------------------------------------------- summary
def available_sources(b, t):
    """Sources whose runs (and controls) all have predictions for this backbone x target."""
    if b in REF_BACKBONES:
        return []
    return [s for s in SOURCES
            if all((OUT / b / t / f"preds_{c}.npz").exists() for c in conditions(b, [s]) if c != "imagenet")]


def _target_info(t, te):
    tab, _ = load_target(t)
    y, g, nc = tab["y"].values, tab["group"].values, n_classes(t)
    yte = {sp: y[te[sp]] for sp in SPLITS}
    gidx = {sp: pd.factorize(g[te[sp]])[0] for sp in SPLITS}
    return y, nc, yte, Boot(yte, gidx, nc)


def analyze(b, t, srcs):
    """Mean balanced accuracy and paired gains with bootstrap CIs, for one backbone x target."""
    S = {c: np.load(OUT / b / t / f"preds_{c}.npz") for c in conditions(b, srcs)}
    te = {sp: S["imagenet"][f"split|{sp}|te"] for sp in SPLITS}
    y, nc, yte, B = _target_info(t, te)

    def chosen(c, sp, k):
        return max(readouts(b), key=lambda r: S[c][f"{r}|{sp}|{k}|val_ba"].mean())

    def cond_for(fam, t_):
        return "imagenet" if fam == "imagenet" else f"{fam}_s{t_}"

    fams = ["imagenet"] + srcs + [ctrl(b) + s for s in srcs]
    modes = ["best", last(b)]
    CB = {}
    for fam in fams:
        for t_ in SEEDS:
            c = cond_for(fam, t_)
            for sp in SPLITS:
                for k in KS:
                    for m in modes:
                        r = chosen(c, sp, k) if m == "best" else m
                        CB[fam, t_, sp, k, m] = (S[c][f"{r}|{sp}|{k}|test_pred"] == yte[sp][None]).mean(0)
    acc = []
    for fam in fams:
        for k in KS:
            for m in modes:
                acc.append(dict(backbone=b, target=t, family=fam, k=k, readout=m,
                                ba=100 * np.mean([B.ba(CB[fam, t_, sp, k, m], sp) for t_ in SEEDS for sp in SPLITS])))
    comps = ([(s, "imagenet", m) for s in srcs for m in modes]
             + [(ctrl(b) + s, "imagenet", m) for s in srcs for m in modes]
             + [(s, ctrl(b) + s, last(b)) for s in srcs]
             + [(s, "food", "best") for s in srcs if s != "food" and "food" in srcs]
             + [(a, "plant", m) for a in srcs if a in ABLATIONS and "plant" in srcs for m in modes])
    gains = []
    for a, c, m in comps:
        for k in KS + ["mean"]:
            ks = KS if k == "mean" else [k]
            gg, lo, hi, cells = B.gain(lambda t_, sp: np.mean([CB[a, t_, sp, kk, m] - CB[c, t_, sp, kk, m] for kk in ks], 0))
            gains.append(dict(backbone=b, target=t, a=a, b=c, readout=m, k=k, gain=100 * gg, lo=100 * lo, hi=100 * hi,
                              cells_pos=int(sum(x > 0 for x in cells)), cells=len(cells)))
    return acc, gains


# ---------------------------------------------------------------- references
def fulldata(b, t):
    from joblib import Parallel, delayed
    tab, _ = load_target(t)
    y, g, nc = tab["y"].values, tab["group"].values, n_classes(t)
    splits = make_splits(y, g, nc)
    d = OUT / b / t
    conds = [c for c in conditions(b) if not c.startswith(ctrl(b)) or c == "imagenet"]
    conds = [c for c in conds if (d / f"feats_{c}.npz").exists() and not (d / f"full_{c}.npz").exists()]
    print(f"{b} {t}: full-data fits for {len(conds)} conditions on {cpus()} cpus", flush=True)
    out = Parallel(n_jobs=cpus(), verbose=2)(delayed(_full_job)(d / f"feats_{c}.npz", b, splits, y) for c in conds)
    for c, res in zip(conds, out):
        np.savez_compressed(d / f"full_{c}.npz", **res)


def full_analyze(b, t, srcs):
    d = OUT / b / t
    conds = ["imagenet"] + [f"{s}_s{t_}" for s in srcs for t_ in SEEDS]
    if not all((d / f"full_{c}.npz").exists() for c in conds):
        return [], []
    te = {sp: np.load(d / "preds_imagenet.npz")[f"split|{sp}|te"] for sp in SPLITS}
    y, nc, yte, B = _target_info(t, te)
    F = {c: np.load(d / f"full_{c}.npz") for c in conds}
    cond_for = lambda fam, t_: "imagenet" if fam == "imagenet" else f"{fam}_s{t_}"
    CB = {(fam, t_, sp): (F[cond_for(fam, t_)][str(sp)] == yte[sp]).astype(float)
          for fam in ["imagenet"] + srcs for t_ in SEEDS for sp in SPLITS}
    acc = [dict(backbone=b, target=t, family=fam,
                ba=100 * np.mean([B.ba(CB[fam, t_, sp], sp) for t_ in SEEDS for sp in SPLITS])) for fam in ["imagenet"] + srcs]
    gains = []
    for s in srcs:
        gg, lo, hi, cells = B.gain(lambda t_, sp: CB[s, t_, sp] - CB["imagenet", t_, sp])
        gains.append(dict(backbone=b, target=t, a=s, gain=100 * gg, lo=100 * lo, hi=100 * hi,
                          cells_pos=int(sum(x > 0 for x in cells)), cells=len(cells)))
    return acc, gains


def ref_analyze(t):
    """Frozen CLIP ViT-B/16 vs each backbone's ImageNet-only model, best readout, paired over splits and test groups."""
    if not (OUT / "clip_b16" / t / "preds_imagenet.npz").exists():
        return []
    S = {b: np.load(OUT / b / t / "preds_imagenet.npz") for b in ALL_BACKBONES
         if (OUT / b / t / "preds_imagenet.npz").exists()}
    te = {sp: S["clip_b16"][f"split|{sp}|te"] for sp in SPLITS}
    y, nc, yte, B = _target_info(t, te)

    def cb(b, sp, k):
        r = max(readouts(b), key=lambda r: S[b][f"{r}|{sp}|{k}|val_ba"].mean())
        return (S[b][f"{r}|{sp}|{k}|test_pred"] == yte[sp][None]).mean(0)

    rows = []
    for k in KS:
        clip_ba = 100 * np.mean([B.ba(cb("clip_b16", sp, k), sp) for sp in SPLITS])
        for b in BACKBONES:
            if b not in S:
                continue
            gg, lo, hi, _ = B.gain(lambda t_, sp: cb("clip_b16", sp, k) - cb(b, sp, k))
            rows.append(dict(target=t, k=k, clip=clip_ba, vs=b,
                             other=100 * np.mean([B.ba(cb(b, sp, k), sp) for sp in SPLITS]),
                             clip_minus=f"{100 * gg:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"))
    return rows


# ---------------------------------------------------------------- cross-pair analysis
def cross_pair(gains, attr, srcs, label):
    """Does attribute distance predict the transfer gain (best readout, mean over k), beyond overall ImageNet-feature
    similarity? Fits OLS with a separate intercept per target. The permutation test permutes the attribute distances
    among the sources within each target, exactly when there are at most N_PERM arrangements, else with N_PERM random
    ones."""
    g = gains[(gains.b == "imagenet") & (gains.readout == "best") & (gains.k == "mean") & gains.a.isin(srcs)]
    lines = []
    rng = np.random.RandomState(0)
    for b in BACKBONES + ["mean"]:
        gg = g if b == "mean" else g[g.backbone == b]
        if gg.empty:
            continue
        gg = gg.groupby(["target", "a"]).gain.mean().reset_index()
        tg = [t for t in TARGETS if set(srcs) <= set(gg[gg.target == t].a)]
        if len(tg) < 2:
            continue
        for kind in ["attr", "attr_nocolour"]:
            rows = [(t, s, gg[(gg.target == t) & (gg.a == s)].gain.item(), attr[kind][s][t], attr["inet"][s][t])
                    for t in tg for s in srcs]
            d = pd.DataFrame(rows, columns=["target", "source", "gain", "attr", "inet"])
            X = np.column_stack([pd.get_dummies(d.target).values.astype(float), d.attr, d.inet])

            def beta(attr_vals):
                Xp = X.copy(); Xp[:, -2] = attr_vals
                return np.linalg.lstsq(Xp, d.gain.values, rcond=None)[0][-2:]

            obs = beta(d.attr.values)
            blocks = [d.index[d.target == t].values for t in tg]
            n_exact = np.prod([float(math.factorial(len(bk))) for bk in blocks])
            if n_exact <= N_PERM:
                perms = list(itertools.permutations(range(len(srcs))))
                combos = itertools.product(perms, repeat=len(tg))
            else:
                combos = ([rng.permutation(len(bk)) for bk in blocks] for _ in range(N_PERM))
            null = []
            for combo in combos:
                vals = d.attr.values.copy()
                for bk, p in zip(blocks, combo):
                    vals[bk] = d.attr.values[bk][list(p)]
                null.append(beta(vals)[0])
            pval = float(np.mean(np.abs(null) >= abs(obs[0]) - 1e-12))
            rho = float(np.mean([d[d.target == t].gain.corr(d[d.target == t].attr, method="spearman") for t in tg]))
            lines.append(dict(set=label, backbone=b, distance=kind, beta_attr=obs[0], beta_inet=obs[1], perm_p=pval,
                              permutations="exact" if n_exact <= N_PERM else f"{N_PERM} random",
                              mean_within_target_spearman=rho, corr_attr_inet=np.corrcoef(d.attr, d.inet)[0, 1],
                              pairs=len(d)))
    return pd.DataFrame(lines)


# ---------------------------------------------------------------- summary
def _txt(df):
    return df.assign(txt=[f"{r.gain:+.1f} [{r.lo:+.1f}, {r.hi:+.1f}] {r.cells_pos}/{r.cells}" for r in df.itertuples()])


def summarize():
    acc, gains, fa, fg, done, abl_done = [], [], [], [], [], []
    for b in BACKBONES:
        for t in TARGETS:
            srcs = available_sources(b, t)
            if not set(MAIN_SOURCES) <= set(srcs):
                continue
            a, g = analyze(b, t, srcs)
            acc += a; gains += g; done.append((b, t))
            if set(ABLATIONS) <= set(srcs):
                abl_done.append((b, t))
            x, y_ = full_analyze(b, t, srcs)
            fa += x; fg += y_
            print("analyzed", b, t, srcs, flush=True)
    acc, gains, fa, fg = pd.DataFrame(acc), pd.DataFrame(gains), pd.DataFrame(fa), pd.DataFrame(fg)
    for name, df in [("accuracy", acc), ("gains", gains), ("full_accuracy", fa), ("full_gains", fg)]:
        df.to_csv(OUT / f"{name}.csv", index=False)
    attr = json.load(open(OUT / "attributes.json")) if (OUT / "attributes.json").exists() else None
    L = ["Leaves to Lesions -- the screen (frozen features), with the PlantVillage ablations, CLIP and the full-data fits "
         "where available.",
         f"Completed backbone x target cells: {len(done)} of {len(BACKBONES) * len(TARGETS)}; with the PlantVillage "
         f"ablations: {len(abl_done)}.",
         f"Sources size-matched to {N_SOURCE} images; {len(SEEDS)} training seeds; {len(SPLITS)} grouped splits; "
         f"k = {KS} x {REPEATS} draws; balanced accuracy (%); 'best' = readout chosen on validation; CIs: 95% "
         "hierarchical bootstrap over seeds, splits and test groups (approximate with 3 seeds). Cells = seed x split "
         "cells with a positive gain. Exploratory for HAM10000 (most lesions seen in earlier audits).", ""]
    if not gains.empty:
        base = (gains.k == "mean")
        g = _txt(gains[base & (gains.b == "imagenet") & (gains.readout == "best") & gains.a.isin(MAIN_SOURCES)])
        L.append("Transfer gain over ImageNet at the best readout, mean over k (points, 95% CI, cells positive):\n")
        L.append(g.pivot_table(index=["backbone", "target"], columns="a", values="txt", aggfunc="first")[MAIN_SOURCES].to_string())
        gl = _txt(gains[base & (gains.b == "imagenet") & (gains.readout != "best")
                        & gains.a.isin(MAIN_SOURCES + [c + s for c in ("bn", "ln") for s in MAIN_SOURCES])])
        L.append("\nAt the last layer, and the normalisation-only controls (bn = BatchNorm statistics, ln = LayerNorm "
                 "affine), vs ImageNet:\n")
        L.append(gl.pivot_table(index=["backbone", "target"], columns="a", values="txt", aggfunc="first").to_string())
        gf = _txt(gains[base & (gains.b == "food") & gains.a.isin(MAIN_SOURCES)])
        L.append("\nSource minus Food-101 at the best readout (is it the domain, or just more training?):\n")
        L.append(gf.pivot_table(index=["backbone", "target"], columns="a", values="txt", aggfunc="first").to_string())
        ab = _txt(gains[base & gains.a.isin(ABLATIONS) & (gains.readout == "best") & gains.b.isin(["imagenet", "plant"])])
        if not ab.empty:
            L.append("\nPlantVillage ablations at the best readout: vs ImageNet and vs intact PlantVillage (gray = no "
                     "colour, blur = no fine dots, shuf = no leaf shape):\n")
            L.append(ab.pivot_table(index=["backbone", "target"], columns=["a", "b"], values="txt", aggfunc="first").to_string())
        L.append("\nMean balanced accuracy (%), best readout:\n")
        bt = acc[acc.readout == "best"].pivot_table(index=["backbone", "target", "k"], columns="family", values="ba")
        L.append(bt[[c for c in ["imagenet"] + SOURCES if c in bt.columns]].round(1).to_string())
    ref = pd.DataFrame([r for t in TARGETS for r in ref_analyze(t)])
    if not ref.empty:
        L.append("\nReference: frozen CLIP ViT-B/16 (400M image-text pairs) vs each backbone's ImageNet-only model, "
                 "best readout (CLIP minus other, points, 95% CI):\n")
        L.append(ref.pivot_table(index=["target", "k"], columns="vs", values="clip_minus", aggfunc="first").to_string())
        L.append(ref.groupby(["target", "k"]).clip.first().round(1).to_string())
    if not fg.empty:
        L.append("\nFull-data ceiling (logistic regression on all training-pool labels, readout s2+s3+s4 / b6+b9+b12): "
                 "source minus ImageNet:\n")
        L.append(_txt(fg).pivot_table(index=["backbone", "target"], columns="a", values="txt", aggfunc="first").to_string())
        L.append(fa.pivot_table(index=["backbone", "target"], columns="family", values="ba").round(1).to_string())
    if attr:
        srcs_attr = [s for s in SOURCES if s in attr["attr"]]
        L.append("\nAttribute distance (Frechet, standardised hand-crafted features; with / without colour) and ImageNet-"
                 "feature Frechet distance, source x target:\n")
        rows = [dict(source=s, target=t, attr=attr["attr"][s][t], attr_nocolour=attr["attr_nocolour"][s][t],
                     inet=attr["inet"][s][t]) for s in srcs_attr for t in TARGETS]
        L.append(pd.DataFrame(rows).pivot(index="source", columns="target").round(1).to_string())
        if not gains.empty:
            cps = [cross_pair(gains, attr, MAIN_SOURCES, "main (4 sources x 3 targets)")]
            if set(ABLATIONS) <= set(srcs_attr) and abl_done:
                cps.append(cross_pair(gains, attr, [s for s in SOURCES if s in srcs_attr], "all (7 sources incl. ablations)"))
            cp = pd.concat(cps)
            if not cp.empty:
                cp.to_csv(OUT / "cross_pair.csv", index=False)
                L.append("\nCross-pair analysis: gain ~ target intercept + attribute distance + ImageNet distance "
                         "(permutation test for the attribute coefficient; exploratory):\n")
                L.append(cp.round(3).to_string(index=False))
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt)
    print(txt)


def train_tasks(sources):
    return [i for i, (b, s, t) in enumerate(itertools.product(BACKBONES, SOURCES, SEEDS)) if s in sources]


# ---------------------------------------------------------------- shared code (common/), with this module's settings
# These call common/ with this module's settings, read at call time (scripts reassign some of them).
def build(b, cond):
    return _B.build(b, cond, OUT / b / "ckpt", SOURCES)


def cache(t):
    _D.cache(t, OUT)


def load_target(t):
    return _D.load_target(t, OUT)


def extract(b, t):
    tab, images = load_target(t)
    _F.extract(b, f"{b} {t}", trained_conditions(b), tab, images, OUT / b / t, build)


def make_splits(y, g, nc):
    return _SP.make_splits(y, g, nc, SPLITS, KS, REPEATS, P.TEST_FRAC)


def evaluate(b, t):
    tab, _ = load_target(t)
    y, g, nc = tab["y"].values, tab["group"].values, n_classes(t)
    _FS.evaluate(b, t, trained_conditions(b), y, nc, make_splits(y, g, nc), OUT / b / t, KS)


class Boot(_BS.Boot):
    """common.bootstrap.Boot over this module's SEEDS and SPLITS as they are when it is built."""
    def __init__(self, yte, gidx, nc, n_boot=_BS.N_BOOT):
        super().__init__(yte, gidx, nc, SEEDS, SPLITS, n_boot)


def _full_job(path, b, splits, y):
    """Full-data ceiling: logistic regression on every training-pool image (inner train + validation), readout cat."""
    import warnings
    from sklearn.linear_model import LogisticRegression
    warnings.filterwarnings("ignore")
    X, dims = readout(np.load(path), "cat", b)
    out = {}
    for sp, split in splits.items():
        pool = np.concatenate([split["itr"], split["val"]])
        mu, sd, scale = _H.shot_scaling(X[pool], dims)
        clf = LogisticRegression(C=Q.C, max_iter=3000, class_weight="balanced").fit((X[pool] - mu) / sd * scale, y[pool])
        out[str(sp)] = clf.predict((X[split["te"]] - mu) / sd * scale).astype(np.int8)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["train", "cache", "extract", "evaluate", "fulldata", "attributes", "summarize",
                                     "tasks"])
    ap.add_argument("--backbone", choices=ALL_BACKBONES)
    ap.add_argument("--source", choices=SOURCES)
    ap.add_argument("--target", choices=TARGETS)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--task", type=int, help="SLURM array index (train: 0-143; extract/evaluate/fulldata: 0-14; cache: 0-2)")
    a = ap.parse_args()
    if a.task is not None:
        if a.step == "train":
            a.backbone, a.source, a.seed = list(itertools.product(BACKBONES, SOURCES, SEEDS))[a.task]
        elif a.step in ("extract", "evaluate", "fulldata"):
            a.backbone, a.target = list(itertools.product(ALL_BACKBONES, TARGETS))[a.task]
        elif a.step == "cache":
            a.target = TARGETS[a.task]
    if a.step == "train":
        train(a.backbone, a.source, a.seed)
    elif a.step == "cache":
        cache(a.target)
    elif a.step == "extract":
        extract(a.backbone, a.target)
    elif a.step == "evaluate":
        evaluate(a.backbone, a.target)
    elif a.step == "fulldata":
        fulldata(a.backbone, a.target)
    elif a.step == "attributes":
        attributes()
    elif a.step == "tasks":
        print(",".join(map(str, train_tasks(ABLATIONS))))
    else:
        summarize()
