"""Does a start trained on synthetic lesions still help after few-shot adaptation on the BSCD-FSL benchmark?

The synthetic-lesion training covers the COIL-100 networks and their checks. From development on, the plan was written
and committed before any run. DINO ViT-S/16 throughout.
  plain   ImageNet DINO (the condition "imagenet")
  ours    plain trained further on synthetic lesion pairs made from COIL-100 views only (pairs_coil_s0-5). It uses the
          pointing experiment's lesion generator v1 and pair loss (synthetic_lesions.py) with the paper's recipe: blocks
          7-12 and the final norm, SGD 0.01, momentum 0.9, no weight decay, clipping at 1, cosine over 840 steps of 32
          pairs. Each seed uses 2,700 views and never a PlantVillage image.
Methods, all under fp16 autocast on the GPU:
  lp      logistic regression on frozen features
  lora    rank-r adapters on the query and value projections, with the head from lp
  ft      full fine-tuning of the last 2 or 6 blocks with the final norm, or of every parameter, with the head from lp
The benchmark episodes are BSCD-FSL's 5-way 1- and 5-shot episodes (15 queries, 600 per target and shot) on ISIC,
ChestX, EuroSAT and CropDisease. Development runs on plain only, on episodes of PAD-UFES-20 (skin, for ISIC) and of CUB
and Places (general, for the other targets). Ours runs with plain's choices. Nothing is tuned or chosen on a target.
PAD-UFES-20 is never reported as a result. The go/no-go check compares plain and ours under full fine-tuning, on 600
new 5-shot episodes each of PAD-UFES-20 and CUB. The final grid runs once, with full fine-tuning as the headline
method. The fourth method, REAP's source stage and frozen test, runs in a separate REAP script in our full codebase. The
paper's linear-probe tests run only with the synthetic-lesion training.
Steps, each run as its own SLURM job:
  Synthetic-lesion training
  make --task s        seed s's synthetic lesions on COIL-100 views (CPU; s in 0-5)
  train --task s       seed s's network, then the learnt check on held-out pairs (GPU)
  features --task i    one network's features on the paper's linear-probe tests (GPU; 0 plain, 1-6 ours)
  evaluate --task i    one network's linear probe on the paper's linear-probe tests (CPU)
  check                ours against the leaves + COIL-100 network on HAM10000 validation -> synthetic_lesion_check.json
                       (CPU)
  Adaptation
  data --task i        one dataset's table, fixed episodes (and the go/no-go check's), 257-px images (CPU; DATA_STEPS)
  crop_check           CropDisease's download against BSCD-FSL's definition -> data/cropdisease_check.json (CPU)
  step0                the step-0, block and adapter checks on one episode, and PAD-UFES-20's lesions (GPU; smoke test)
  dev_a --task i       plain x development domain x shot x method: every grid configuration (GPU; DEV_A_TASKS)
  choose               plain's choices per domain group, shot and method -> dev.json, dev_choices.csv (CPU)
  gonogo --task i      one network x go/no-go domain: full fine-tuning as chosen, on its own episodes (GPU; GNG_TASKS)
  gonogo_summary       -> gonogo.csv and gonogo.txt (CPU)
  final_a --task i     one network x target x shot: lp, lora and ft as chosen (GPU; FINAL_A_TASKS)
  summarize            -> benchmark_results.csv, leaderboard.csv (against docs/leaderboard.md), verdict.json and
                       summary.txt, with REAP's rows from reap/reap.csv once the REAP script has written it (CPU)
Checkpoints, features and predictions on HAM10000, DDR and ISIC go where the pipeline keeps them (screen.OUT,
credibility.OUT). Everything else goes to outputs/benchmark. BENCHMARK_QUICK=1 runs a smoke test into
outputs/benchmark/quick.
"""
import argparse
import copy
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

_PILOT = Path(__file__).resolve().parents[1]
for _d in ([_PILOT.parent] + sorted((_PILOT.parent / "archive" / "paper-v1").glob("[1-9]_*"))
           + [_PILOT / "10_classifiers", _PILOT / "15_pointing"]):
    sys.path.insert(0, str(_d))                  # common/, the paper's nine experiments, then 10 and 15
QUICK = os.environ.get("BENCHMARK_QUICK") == "1"
import first_pilot as pilot  # noqa: E402
import corrected_pilot  # noqa: E402
import source_screen as screen  # noqa: E402
import credibility_checks as credibility  # noqa: E402
import pilot_fine_tuning  # noqa: E402  (Head, fit_head, augment, the recipe's constants, gpu_setup)
import fine_tuning  # noqa: E402  (Net, chosen_readout)
import classifiers  # noqa: E402  (code_version, save, target, isic_units)
import synthetic_lesions as lesions  # noqa: E402  (the lesion generator, PairSet, Heads, step_loss, the reserved
#                                                  images, other_splits, LesionBootWithDraws, t_interval, level)

OUT = pilot.ROOT / "outputs" / "benchmark" / ("quick" if QUICK else "")
B = "vits_dino"
FAM, P2 = "pairs_coil", "pairs_synth"          # ours, and the pointing experiment's leaves + COIL-100 network
SEEDS, DEV_SEEDS = [0, 1, 2, 3, 4, 5], [0, 1, 2]
NETS = ["imagenet"] + [f"{FAM}_s{s}" for s in SEEDS]
TESTS_C = ["ham", "isic_test", "ddr", "hyperkvasir6"]
N_COIL = 16 if QUICK else 2700                 # 27 views of each of the 100 objects
SYNTH_SEED = 50                                # seed s's lesions and fills are drawn with SYNTH_SEED + s
STEPS = 2 if QUICK else lesions.STEPS              # 840
BATCH_PAIRS = 8 if QUICK else lesions.BATCH_PAIRS  # 32 pairs, 64 images
LEARNT, CHECK_GAP = lesions.LEARNT, 1.0            # 0.75 held-out pair accuracy, and HAM10000 validation within
#                                                    1.0 point of the leaves + COIL-100 network
CKPT = OUT / "ckpt" if QUICK else screen.OUT / B / "ckpt"

# the benchmark episodes and their development domains
DATA_A = ["isic", "chestx", "eurosat", "cropdisease"]
DATA_DEV = ["cub", "places", "pad"]            # development only, never reported as a result
DATASETS = DATA_A + DATA_DEV                   # an episode set's seed depends on the dataset's place here
DATA_STEPS = ["isic", "cub", "pad"] if QUICK else DATASETS
DEV_DOMAINS = ["cub", "pad"] if QUICK else DATA_DEV
GROUPS = {"skin": ["pad"], "general": ["cub"] if QUICK else ["cub", "places"]}   # where each group's choices are made
GROUP_OF = {"isic": "skin", "pad": "skin", "chestx": "general", "eurosat": "general", "cropdisease": "general",
            "cub": "general", "places": "general"}
SHOTS, WAYS, QUERIES = [1, 5], 5, 15
N_EPISODES = {**dict.fromkeys(DATA_A, 3 if QUICK else 600), **dict.fromkeys(DATA_DEV, 3 if QUICK else 100)}
GNG_DOMAINS, GNG_SHOT, N_GNG = ["pad", "cub"], 5, 3 if QUICK else 600   # the go/no-go check's own episodes
SQUARE, CROP = 257, 224                        # BSCD-FSL's test transform: square resize to 257, centre crop 224
OFF = int(round((SQUARE - CROP) / 2))          # 16, where torchvision's CenterCrop puts it
N_NCT = 500                                    # NCT-CRC-HE-100K patches per class (synthetic-lesion training)
DATA_OUT = OUT / "data"
BLOCKS = (2, 6, 12)                            # full fine-tuning trains the last 2 or 6 blocks, or every parameter
FT_GRID = ([dict(lr=1e-2, steps=2, blocks=b) for b in (2, 12)] if QUICK else
           [dict(lr=lr, steps=st, blocks=b) for lr in (1e-3, 3e-3, 1e-2, 3e-2) for st in (25, 50, 100) for b in BLOCKS])
LORA_GRID = ([dict(rank=4, lr=1e-2, steps=2)] if QUICK else    # SGD: adapters need larger rates than full weights
             [dict(rank=r, lr=lr, steps=100) for r in (4, 16) for lr in (1e-2, 3e-2, 1e-1)])
GRIDS = {"ft": FT_GRID, "lora": LORA_GRID}
METHODS = ["lp", "lora", "ft"]
DEV_STARTS = ["imagenet"]                      # development runs on plain only; ours takes plain's choices
DEV_A_TASKS = [(n, s, m) for n in DEV_DOMAINS for s in SHOTS for m in GRIDS]
GNG_TASKS = [(c, n) for c in (["imagenet"] if QUICK else NETS) for n in GNG_DOMAINS]
FINAL_A_TASKS = [("imagenet", "isic", 5)] if QUICK else [(c, n, s) for c in NETS for n in DATA_A for s in SHOTS]
HEADLINE = "ft"                                # the method behind the headline number, fixed before the final grid

REAP, REAP_H, REAP_1 = 55.28, 0.24, 38.67      # the number to beat (docs/leaderboard.md), its half-width, and 1-shot
LEADERBOARD = pilot.ROOT / "docs" / "leaderboard.md"
LB_NAMES = {"ChestX": "chestx", "ISIC": "isic", "EuroSAT": "eurosat", "CropDisease": "cropdisease"}
CONFIG = dict(quick=QUICK, backbone=B, family=FAM, seeds=SEEDS, dev_seeds=DEV_SEEDS, n_coil=N_COIL,
              synth_seed=SYNTH_SEED, steps=STEPS, batch_pairs=BATCH_PAIRS, lr=pilot.LR, learnt=LEARNT,
              check_gap=CHECK_GAP, tests_c=TESTS_C, data_a=DATA_A, data_dev=DATA_DEV, groups=GROUPS, shots=SHOTS,
              ways=WAYS, queries=QUERIES, n_episodes=N_EPISODES, gonogo=dict(domains=GNG_DOMAINS, shot=GNG_SHOT, n=N_GNG),
              square=SQUARE, crop=CROP, n_nct=N_NCT, ft_grid=FT_GRID, lora_grid=LORA_GRID, headline=HEADLINE,
              reap=(REAP, REAP_H, REAP_1), precision="fp16 autocast with loss scaling")


# ---------------------------------------------------------------- files
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(step, task=None, **extra):
    """The code version, the hashes of this file and of source_screen.py, the configuration and the environment, saved with
    every output."""
    from importlib import metadata
    versions = {}
    for p in ["numpy", "scipy", "scikit-learn", "scikit-image", "pandas", "torch", "torchvision", "timm", "joblib"]:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    m = dict(step=step, task=task, code_version=classifiers.code_version(), script_sha256=sha256(__file__),
             screen_sha256=sha256(screen.__file__), config=CONFIG, python=platform.python_version(), packages=versions,
             host=platform.node(), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
             slurm_job=os.environ.get("SLURM_JOB_ID"), slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"),
             finished=time.strftime("%Y-%m-%d %H:%M:%S"), **extra)
    d = OUT / "manifests"
    d.mkdir(parents=True, exist_ok=True)
    name = step if task is None else f"{step}_{task}"
    json.dump(m, open(d / f"{name}.json", "w"), indent=1, default=str)


def kind(c):
    return "plain" if c == "imagenet" else "ours"


# ---------------------------------------------------------------- synthetic-lesion training on COIL-100 views
def coil_bases(seed):
    """A seed's clean COIL-100 views, spread as evenly over the 100 objects as the unreserved views allow (27 each).
    Views the pointing experiment reserves (its probe set and learnt check) are never drawn."""
    rng, C = random.Random(seed), lesions.coil_files()
    view = {o: lesions.unreserved(C[o]) for o in sorted(C)}
    cnt = lesions.quota({o: len(v) for o, v in view.items()}, N_COIL, rng)
    return [(f, "coil") for o in view for f in rng.sample(view[o], cnt[o])]


def synth_set(name, bases, seed):
    """Lesioned copies of `bases` [(path, kind)] -> outputs/benchmark/synth/<name>/ (NNNN.png, NNNN_mask.png,
    index.csv). Each lesion is filled from another of the bases. The same as lesions.synth_set, but in this experiment's
    folder."""
    d = OUT / "synth" / name
    if (d / "index.csv").exists():
        print("exists:", d); return
    d.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    files = [f for f, _ in bases]
    others = []
    for f in files:
        o = f
        while o == f:
            o = files[rng.randrange(len(files))]
        others.append(o)
    jobs = [(f, o, str(d / f"{i:04d}.png"), seed * 100000 + i) for i, (f, o) in enumerate(zip(files, others))]
    with Pool(screen.cpus()) as pool:
        recs = pool.map(lesions._lesion_job, jobs, chunksize=8)
    df = pd.DataFrame([dict(i=i, base=f, kind=k, other=o, lesioned=j[2], mask=lesions.mask_path(j[2]), **r)
                       for i, ((f, k), o, j, r) in enumerate(zip(bases, others, jobs, recs))])
    df.to_csv(d / "index.tmp.csv", index=False)
    os.replace(d / "index.tmp.csv", d / "index.csv")
    share = df.lesion_px / df.object_px
    print(f"{name}: {len(df)} lesions; lesion share of the object median {share.median():.3f} "
          f"[{share.quantile(0.05):.3f}, {share.quantile(0.95):.3f}]; below 16 px: {(df.lesion_px < 16).sum()}",
          flush=True)
    lesions.sheet(df.head(12), OUT / "sheets" / f"lesions_{name}.jpg")


def make(task):
    t0 = time.time()
    bases = coil_bases(SEEDS[task])
    assert len({f for f, _ in bases}) == N_COIL and all("coil" in f.lower() for f, _ in bases)
    synth_set(f"c{SEEDS[task]}", bases, SYNTH_SEED + SEEDS[task])
    manifest("make", task, seconds=round(time.time() - t0))


def learnt(m, heads, dev):
    """Held-out pair accuracy on the pointing experiment's 200 reserved synthetic pairs (100 on leaves, 100 on
    COIL-100 views), and on the 100 on COIL-100 views alone. These pairs are scored only, never trained on."""
    m.eval(), heads.eval()
    H, out = lesions.held_sets(), {}
    for name in ("synthetic", "coil"):
        pairs = H[name]
        sa = lesions.scores(m, heads, [p["normal"] for p in pairs], dev)
        sb = lesions.scores(m, heads, [p["abnormal"] for p in pairs], dev)
        out[name] = float((sb > sa).mean())
    print("held-out pair accuracy:", {k: round(v, 3) for k, v in out.items()}, flush=True)
    return out


def train(task):
    """The pointing experiment's training loop with the pair loss alone. lesions.train would give a family it doesn't
    know a source head."""
    seed = SEEDS[task]
    ck = CKPT / f"{FAM}_s{seed}.pt"
    if ck.exists():
        print("exists:", ck); return
    t0 = time.time()
    pilot.seed_all(seed)
    S = pd.read_csv(OUT / "synth" / f"c{seed}" / "index.csv")
    pairs = [dict(normal=r.base, abnormal=r.lesioned, mask=r.mask, source=2) for r in S.itertuples()]
    dl = torch.utils.data.DataLoader(lesions.PairSet(pairs), batch_size=BATCH_PAIRS, shuffle=True, drop_last=True,
                                     num_workers=min(6, screen.cpus()), persistent_workers=True)
    assert QUICK or (len(pairs) == 2700 and len(dl) == 84), (len(pairs), len(dl))
    print(f"{FAM} seed {seed}: {len(pairs)} pairs on COIL-100 views, pair loss", flush=True)
    dev = pilot.device()
    m = screen.imagenet_model(B)
    trainable = tuple(f"blocks.{i}." for i in corrected_pilot.VIT_TRAINED) + ("norm.",)
    for n_, p in m.named_parameters():
        p.requires_grad = n_.startswith(trainable)
    heads = lesions.Heads()
    m.to(dev), heads.to(dev)
    params = [p for p in m.parameters() if p.requires_grad] + list(heads.parameters())
    opt = torch.optim.SGD(params, lr=pilot.LR, momentum=0.9, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    log, done = [], 0
    while done < STEPS:
        m.train(), heads.train()
        t1, tot, hit, n = time.time(), 0.0, 0, 0
        for batch in dl:
            if done == STEPS:
                break
            loss, h_, n_ = lesions.step_loss(m, heads, batch, "pair", 0.0, dev)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(), sched.step()
            done += 1
            tot, hit, n = tot + loss.item() * n_, hit + h_, n + n_
        log.append(dict(epoch=len(log) + 1, loss=tot / max(n, 1), train_acc=hit / max(n, 1), sec=round(time.time() - t1)))
        print(log[-1], flush=True)
    held = learnt(m, heads, dev)
    heads_dir = OUT / "heads"
    for d in (ck.parent, heads_dir):
        d.mkdir(parents=True, exist_ok=True)
    torch.save(heads.state_dict(), heads_dir / f"{ck.stem}.pt")
    json.dump(dict(backbone=B, source=FAM, seed=seed, n=2 * len(pairs), pairs=len(pairs), steps=STEPS, lr=pilot.LR,
                   objective=dict(loss="pair", patch=False, source=False, adversarial=False), log=log,
                   heldout_pair_acc=held, sec=round(time.time() - t0)), open(ck.with_suffix(".json"), "w"), indent=1)
    tmp = ck.with_suffix(".tmp")
    torch.save({k: v.cpu() for k, v in m.state_dict().items() if not k.startswith("head.")}, tmp)   # as screen.train
    os.replace(tmp, ck)
    if QUICK:                                      # the checkpoint loads the way screen.build loads one
        m2 = screen.imagenet_model(B)
        missing, unexpected = m2.load_state_dict(torch.load(ck, map_location="cpu"), strict=False)
        assert not unexpected and set(missing) == screen.head_keys(B), (missing, unexpected)
        print("quick: the checkpoint loads, with only the head keys missing", flush=True)
    manifest("train", task, network=ck.stem, heldout_pair_acc=held, seconds=round(time.time() - t0))


# ---------------------------------------------------------------- the paper's linear-probe tests
def features(task):
    c = NETS[task]
    t0 = time.time()
    for t in TESTS_C:                                  # existing files are kept (plain's, mostly)
        credibility.extract_conditions(B, t, [c])
    print(f"{c}: features in {time.time() - t0:.0f}s", flush=True)
    manifest("features", task, network=c, seconds=round(time.time() - t0))


def preds_path(t, c):
    """The path of network c's predictions on t. HAM10000's and DDR's are where the pipeline keeps them, and
    HyperKvasir's are in outputs/benchmark, even in a smoke test."""
    return (screen.OUT / B if t in screen.TARGETS else pilot.ROOT / "outputs" / "benchmark" / B) / t / f"preds_{c}.npz"


def evaluate_other(t, cs):
    """credibility.evaluate_conditions for HyperKvasir (k = 5 and 20), saved in outputs/benchmark with the validation
    accuracies."""
    from joblib import Parallel, delayed
    y, _, nc, splits = lesions.other_splits(t)
    todo = [c for c in cs if not preds_path(t, c).exists()]
    jobs = [(c, r) for c in todo for r in screen.readouts(B)]
    out = Parallel(n_jobs=screen.cpus(), verbose=1)(
        delayed(screen._eval_job)(credibility.target_dir(B, t) / f"feats_{c}.npz", B, c, r, splits, y, nc, lesions.OTHER_KS)
        for c, r in jobs)
    per = {}
    for c, res in out:
        per.setdefault(c, {}).update(res)
    for c, res in per.items():
        for sp in screen.SPLITS:
            res[f"split|{sp}|te"] = splits[sp]["te"]
        classifiers.save(preds_path(t, c), res)


def evaluate(task):
    c = NETS[task]
    t0 = time.time()
    for t in ("ham", "ddr"):
        credibility.evaluate_conditions(B, t, [c])
    credibility.external(B, [c])                                # ISIC 2018 test, from the HAM10000 shots
    evaluate_other("hyperkvasir6", [c])
    manifest("evaluate", task, network=c, seconds=round(time.time() - t0))


# ---------------------------------------------------------------- the synthetic-lesion check
def val_mean(c):
    """Mean HAM10000 validation balanced accuracy (%) over splits and k, at the readout the paper's rule picks for each
    split and k (the best mean over the 20 draws)."""
    S = np.load(screen.OUT / B / "ham" / f"preds_{c}.npz")
    return 100 * float(np.mean([max(S[f"{r}|{sp}|{k}|val_ba"].mean() for r in screen.readouts(B))
                                for sp in screen.SPLITS for k in screen.KS]))


def check():
    """The synthetic-lesion check. Ours must come within CHECK_GAP of the leaves + COIL-100 network on HAM10000
    validation (mean over k and seeds 0-2), and every seed must pass the learnt check. Otherwise the job chain stops.
    The smoke test scores the leaves + COIL-100 network against itself, only to run the code."""
    fam = P2 if QUICK else FAM
    ours = {s: val_mean(f"{fam}_s{s}") for s in DEV_SEEDS}
    p2 = {s: val_mean(f"{P2}_s{s}") for s in DEV_SEEDS}
    gap = float(np.mean(list(ours.values())) - np.mean(list(p2.values())))
    held = {} if QUICK else {s: json.load(open(CKPT / f"{FAM}_s{s}.json"))["heldout_pair_acc"] for s in SEEDS}
    learnt_ok = all(v["synthetic"] >= LEARNT for v in held.values())
    res = dict(ours_val=ours, p2_val=p2, ours_mean=float(np.mean(list(ours.values()))),
               p2_mean=float(np.mean(list(p2.values()))), gap=gap, gap_ok=gap >= -CHECK_GAP, heldout_pair_acc=held,
               learnt_ok=learnt_ok, passes=bool(gap >= -CHECK_GAP and learnt_ok))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "synthetic_lesion_check.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)
    manifest("check", passes=res["passes"])
    if not res["passes"]:
        raise SystemExit("The synthetic-lesion check failed (see synthetic_lesion_check.json)")


# ---------------------------------------------------------------- the data of the benchmark and development domains
def table(name):
    """(path, y) and the class names of a dataset. BSCD-FSL's targets are read as its loaders read them:
      isic          the 10,015 ISIC 2018 Task 3 training images
      chestx        every single-label image of its 7 classes (25,848)
      eurosat       27,000 RGB images
      cropdisease   its Kaggle copy's dataset/train
    The development domains:
      cub           CUB-200-2011's 11,788 images
      places        Places365's 36,500 validation images (256 px)
      pad           PAD-UFES-20's 2,298 photos of 6 diagnoses, with a group column for their 1,891 lesions
    nct, as in the synthetic-lesion training, is 500 NCT-CRC-HE-100K patches per class."""
    D = pilot.DATA
    if name == "isic":
        return screen.target_table("ham")[["path", "y"]].reset_index(drop=True), list(pilot.HAM_CLASSES)
    if name == "pad":
        raw = D / "pad_ufes20" / "raw"
        meta = pd.read_csv(next(raw.rglob("metadata.csv")))
        files = {p.name: p for p in raw.rglob("*.png")}
        key = lambda i: i if i in files else f"{i}.png"
        meta = meta[[key(i) in files for i in meta.img_id]].sort_values("img_id")
        classes = sorted(meta.diagnostic.unique())
        lesion = meta.patient_id + "_" + meta.lesion_id.astype(str)   # lesion ids repeat across patients
        return (pd.DataFrame(dict(path=[str(files[key(i)]) for i in meta.img_id],
                                  y=[classes.index(c) for c in meta.diagnostic], group=lesion.values)), classes)
    if name == "chestx":
        d = pd.read_csv(D / "cxr" / "Data_Entry_2017_v2020.csv")
        d = d[d["Finding Labels"].isin(screen.CXR_CLASSES)]
        d = d[[(D / "cxr" / "images" / f).exists() for f in d["Image Index"]]].sort_values("Image Index")
        return (pd.DataFrame(dict(path=[str(D / "cxr" / "images" / f) for f in d["Image Index"]],
                                  y=[screen.CXR_CLASSES.index(f) for f in d["Finding Labels"]])), list(screen.CXR_CLASSES))
    if name == "places":
        root = D / "places365"
        lab = pd.read_csv(root / "places365_val.txt", sep=" ", header=None, names=["file", "y"])
        cats = [line.split()[0] for line in open(root / "categories_places365.txt")]
        return pd.DataFrame(dict(path=[str(root / "val_256" / f) for f in lab["file"]], y=lab["y"])), cats
    if name == "eurosat":
        root = next((p.parent for p in sorted((D / "eurosat").rglob("AnnualCrop")) if p.is_dir()), D / "eurosat" / "?")
    else:
        root = {"cropdisease": D / "cropdisease" / "dataset" / "train", "cub": D / "cub" / "CUB_200_2011" / "images",
                "nct": D / "nct" / "NCT-CRC-HE-100K"}[name]
    if not root.is_dir():
        raise SystemExit(f"{name}: {root} is missing" + (" (BSCD-FSL's CropDisease, the Kaggle download of "
                                                         "saroz014/plant-disease)" if name == "cropdisease" else ""))
    classes = screen._class_dirs(root)
    df = pd.DataFrame([(str(p), i) for i, c in enumerate(classes) for p in sorted((root / c).iterdir())
                       if p.suffix.lower() in screen.IMG_EXTS], columns=["path", "y"])
    if name == "nct":
        rng = np.random.RandomState(0)
        df = pd.concat([g.iloc[np.sort(rng.choice(len(g), N_NCT, replace=False))] for _, g in df.groupby("y")])
    return df.reset_index(drop=True), classes


def draw_episodes(y, n, shot, seed, groups=None):
    """n episodes. Each takes WAYS classes among those with at least shot + QUERIES images, then shot + QUERIES images
    of each, without replacement. With groups (PAD-UFES-20's lesions), a class needs shot + QUERIES groups and each
    image comes from a different one, so the support set and the queries never share a lesion. Without groups, the
    draws are the same as in the synthetic-lesion training. Returns the classes (n x WAYS), and the support (n x WAYS x
    shot) and query (n x WAYS x QUERIES) row indices."""
    rng = np.random.RandomState(seed)
    size = (lambda c: (y == c).sum()) if groups is None else (lambda c: len(np.unique(groups[y == c])))
    ok = [c for c in np.unique(y) if size(c) >= shot + QUERIES]
    cls = np.empty((n, WAYS), int)
    sup, qry = np.empty((n, WAYS, shot), int), np.empty((n, WAYS, QUERIES), int)
    for e in range(n):
        cls[e] = rng.choice(ok, WAYS, replace=False)
        for j, c in enumerate(cls[e]):
            idx = np.flatnonzero(y == c)
            if groups is None:
                pick = rng.choice(idx, shot + QUERIES, replace=False)
            else:
                gs = rng.choice(np.unique(groups[idx]), shot + QUERIES, replace=False)
                pick = np.array([rng.choice(idx[groups[idx] == g]) for g in gs])
            sup[e, j], qry[e, j] = pick[:shot], pick[shot:]
    return cls, sup, qry


def repeated_groups(ep, groups):
    """How many episodes (development and go/no-go) show a group, here a lesion, more than once."""
    n = 0
    for k in [k for k in ep if k.startswith(("sup", "gsup"))]:
        for s_, q_ in zip(ep[k], ep[k.replace("sup", "qry")]):
            g = groups[np.concatenate([s_.ravel(), q_.ravel()])]
            n += len(np.unique(g)) < len(g)
    return n


def _square(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB").resize((SQUARE, SQUARE), Image.BILINEAR)).transpose(2, 0, 1)


def data(task):
    """One dataset's table, its fixed episodes, and the images any episode uses at SQUARE x SQUARE px (BSCD-FSL's
    square resize). N_EPISODES episodes per shot are drawn once, with a seed that depends only on the dataset and the
    shot. A go/no-go domain also gets N_GNG GNG_SHOT-shot episodes of its own (gsup/gqry), with seeds from 8000."""
    t0 = time.time()
    name = DATA_STEPS[task]
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    tab, classes = table(name)
    tab.to_csv(DATA_OUT / f"{name}_index.csv", index=False)
    json.dump(classes, open(DATA_OUT / f"{name}_classes.json", "w"))
    y, ep = tab["y"].values, {}
    groups = tab["group"].values if "group" in tab else None
    for shot in SHOTS:
        cls, sup, qry = draw_episodes(y, N_EPISODES[name], shot, 7000 + 100 * DATASETS.index(name) + shot, groups)
        ep.update({f"cls{shot}": cls, f"sup{shot}": sup, f"qry{shot}": qry})
    if name in GNG_DOMAINS:
        cls, sup, qry = draw_episodes(y, N_GNG, GNG_SHOT, 8000 + 100 * DATASETS.index(name) + GNG_SHOT, groups)
        ep.update({f"gcls{GNG_SHOT}": cls, f"gsup{GNG_SHOT}": sup, f"gqry{GNG_SHOT}": qry})
    if groups is not None:
        assert repeated_groups(ep, groups) == 0, f"{name}: an episode shows a lesion twice"
    np.savez(DATA_OUT / f"{name}_episodes.npz", **ep)
    rows = np.unique(np.concatenate([v.ravel() for k, v in ep.items() if "sup" in k or "qry" in k]))
    np.save(DATA_OUT / f"{name}_rows.npy", rows)
    dst = DATA_OUT / f"{name}_u8.npy"
    if not (dst.exists() and np.load(dst, mmap_mode="r").shape[0] == len(rows)):
        tmp = DATA_OUT / f"{name}_u8.tmp.npy"
        arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(rows), 3, SQUARE, SQUARE))
        with Pool(screen.cpus()) as pool:
            for i, a in enumerate(pool.imap(_square, tab["path"].values[rows], chunksize=32)):
                arr[i] = a
        arr.flush()
        del arr
        os.replace(tmp, dst)
    n_gng = N_GNG if name in GNG_DOMAINS else 0
    print(f"{name}: {len(tab)} images, {len(classes)} classes; {len(rows)} in {N_EPISODES[name]} episodes a shot and "
          f"{n_gng} go/no-go episodes; {time.time() - t0:.0f}s", flush=True)
    manifest("data", task, dataset=name, n_images=len(tab), n_classes=len(classes), n_cached=len(rows),
             n_gonogo=n_gng, episodes_sha256=sha256(DATA_OUT / f"{name}_episodes.npz"), seconds=round(time.time() - t0))


def crop_check():
    """CropDisease's Kaggle download against BSCD-FSL's definition. BSCD-FSL's loader reads dataset/train with
    ImageFolder, which gives 38 classes and 43,456 images. Writes data/cropdisease_check.json with every class's count,
    and exits on a mismatch."""
    root = pilot.DATA / "cropdisease" / "dataset" / "train"
    if not root.is_dir():
        raise SystemExit(f"{root} is missing (the Kaggle download of saroz014/plant-disease)")
    classes = screen._class_dirs(root)
    counts = {c: sum(p.suffix.lower() in screen.IMG_EXTS for p in (root / c).iterdir()) for c in classes}
    res = dict(root=str(root), n_classes=len(classes), n_images=sum(counts.values()),
               expected=dict(n_classes=38, n_images=43456), counts=counts)
    res["matches"] = (res["n_classes"], res["n_images"]) == (38, 43456)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(DATA_OUT / "cropdisease_check.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "counts"}), flush=True)
    manifest("crop_check", matches=res["matches"], n_classes=res["n_classes"], n_images=res["n_images"])
    if not res["matches"]:
        raise SystemExit("CropDisease does not match BSCD-FSL's definition; see data/cropdisease_check.json")


@functools.lru_cache(None)
def episode_files(name):
    E = dict(np.load(DATA_OUT / f"{name}_episodes.npz"))
    rows = np.load(DATA_OUT / f"{name}_rows.npy")
    pos = np.full(int(rows.max()) + 1, -1)
    pos[rows] = np.arange(len(rows))
    return E, pos, np.load(DATA_OUT / f"{name}_u8.npy", mmap_mode="r")


# ---------------------------------------------------------------- LoRA adapters
class Delta(nn.Module):
    """A projection weight W (out x in) reparametrized as W + (alpha / r) B A. A starts as nn.Linear's weights do
    (Kaiming uniform), B at zero, so the network starts unchanged."""

    def __init__(self, n_out, n_in, r, alpha):
        super().__init__()
        self.scale = alpha / r
        self.a = nn.Parameter(torch.empty(r, n_in))
        self.b = nn.Parameter(torch.zeros(n_out, r))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, w):
        return w + self.scale * (self.b @ self.a)


class QV(nn.Module):
    """A fused qkv weight (3e x in): a Delta on its query rows and one on its value rows, none on the keys."""

    def __init__(self, e, n_in, r, alpha):
        super().__init__()
        self.q, self.v = Delta(e, n_in, r, alpha), Delta(e, n_in, r, alpha)

    def forward(self, w):
        e = w.shape[0] // 3
        return torch.cat([self.q(w[:e]), w[e:2 * e], self.v(w[2 * e:])])


def add_lora(m, r):
    """Rank-r adapters (alpha = r) on the query and value projections of every attention block; everything else in m
    is frozen. Returns the adapters' parameters."""
    from torch.nn.utils import parametrize
    for p in m.parameters():
        p.requires_grad = False
    for blk in m.blocks:
        a = blk.attn.qkv                        # initialized on the CPU (seeded), then moved to the weight's device
        qv = QV(a.out_features // 3, a.in_features, r, r).to(a.weight.device, a.weight.dtype)
        parametrize.register_parametrization(a, "weight", qv)
    return [p for p in m.parameters() if p.requires_grad]


# ---------------------------------------------------------------- adaptation
def amp(dev):
    """fp16 autocast on the GPU, for every method and start alike, and nothing on the CPU."""
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev.type == "cuda")


def center(u8):
    """The benchmark episodes' evaluation input: the centre 224 px of SQUARE-px uint8 images on the device,
    ImageNet-normalized."""
    x = u8[:, :, OFF:OFF + CROP, OFF:OFF + CROP].float().div_(255)
    return (x - pilot_fine_tuning.MEAN.to(x.device)) / pilot_fine_tuning.STD.to(x.device)


@torch.no_grad()
def cls_feats(m, x, bs=128):
    """The CLS token after the final norm (screen.stage_features' "cls"), as float32 numpy."""
    m.eval()
    with amp(x.device):
        return torch.cat([screen.stage_features(B, m, x[s:s + bs])["cls"].float() for s in range(0, len(x), bs)]).cpu().numpy()


@torch.no_grad()
def predict(net, x, bs=128):
    net.eval()
    with amp(x.device):
        return torch.cat([net(x[s:s + bs]).float().argmax(1) for s in range(0, len(x), bs)]).cpu().numpy()


@torch.no_grad()
def predict_u8(net, images, idx, dev, bs=256):
    """Predictions for the pipeline's 224-px evaluation images (uint8), ImageNet-normalized."""
    net.eval()
    out = []
    for s in range(0, len(idx), bs):
        x = torch.from_numpy(np.array(images[idx[s:s + bs]])).to(dev).float().div_(255)
        with amp(dev):
            out.append(net((x - pilot_fine_tuning.MEAN.to(dev)) / pilot_fine_tuning.STD.to(dev)).float().argmax(1).cpu())
    return torch.cat(out).numpy().astype(np.int8)


def cfg_name(m, cfg):
    return m if m == "lp" else "|".join([m] + [f"{k}{v:g}" for k, v in cfg.items()])


def method_params(net, m, cfg, seed):
    """Set which parameters train, and return them. Full fine-tuning trains the last cfg["blocks"] blocks, the final
    norm and the head, or every parameter when that is all the blocks (the default). LoRA trains the adapters (seeded,
    so every start gets the same ones) and the head."""
    if m == "lora":
        torch.manual_seed(seed)
        return add_lora(net.backbone, cfg["rank"]) + list(net.head.parameters())
    n = len(net.backbone.blocks)
    k = cfg.get("blocks", n)
    trainable = tuple(f"backbone.blocks.{i}." for i in range(n - k, n)) + ("backbone.norm.", "head.")
    for name, p in net.named_parameters():
        p.requires_grad = k == n or name.startswith(trainable)
    return [p for p in net.parameters() if p.requires_grad]


def adapt(net, params, batches, steps, lr, dev):
    """The paper's fine-tuning loop (pilot_fine_tuning.finetune), under fp16 autocast with loss scaling. SGD with
    momentum 0.9 and weight decay 1e-4, cosine over `steps`, gradient clipping at 1. batches() returns one batch (x, y)
    on the device."""
    if steps == 0:
        return
    opt = torch.optim.SGD(params, lr=lr, momentum=pilot_fine_tuning.MOMENTUM, weight_decay=pilot_fine_tuning.WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 10, enabled=dev.type == "cuda")   # short runs: few skips
    lossf = nn.CrossEntropyLoss()
    for step in range(steps):
        net.train()
        x, yb = batches()
        with amp(dev):
            loss = lossf(net(x).float(), yb)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step + 1}")
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, pilot_fine_tuning.CLIP)
        scaler.step(opt)
        scaler.update()
        sched.step()


def augment_all(u8, dev):
    """pilot_fine_tuning.augment on every support image, in order: a random resized crop (scale 0.5-1, ratio 3/4-4/3) to
    224 px and a flip with probability 0.5, from the global torch generator."""
    return pilot_fine_tuning.augment(list(u8), torch.arange(len(u8)), dev)


# ---------------------------------------------------------------- the benchmark episodes
def episode_run(c, name, shot, plan, dst, ep=""):
    """Every episode of one dataset and shot, from start c -> dst.

    In each episode, the linear probe is fit on the support set's CLS tokens. Each (label, method, configuration) of
    `plan` then starts from c's weights with that probe as its head, trains on the support images (all of them at
    every step, freshly cropped) and is scored on the queries. The crops and the adapters are seeded by episode, so
    every start gets the same ones. ep="g" runs the go/no-go check's own episodes. dst holds the accuracy per episode
    for each label ("lp" too), and which runs diverged (scored at chance)."""
    if dst.exists():
        print("exists:", dst); return
    t0 = time.time()
    dev = pilot_fine_tuning.gpu_setup()
    E, pos, cache = episode_files(name)
    sup, qry = E[f"{ep}sup{shot}"], E[f"{ep}qry{shot}"]
    n = len(sup)
    base = screen.build(B, c).to(dev).eval()
    labels = ["lp"] + [lab for lab, _, _ in plan]
    acc = {k: np.full(n, np.nan) for k in labels}
    div = {k: np.zeros(n, bool) for k in labels}
    ys, yq = np.repeat(np.arange(WAYS), shot), np.repeat(np.arange(WAYS), QUERIES)
    ysd = torch.as_tensor(ys, device=dev)
    for e in range(n):
        su = torch.from_numpy(np.stack([cache[pos[i]] for i in sup[e].ravel()])).to(dev)
        xq = center(torch.from_numpy(np.stack([cache[pos[i]] for i in qry[e].ravel()])).to(dev))
        Xq = cls_feats(base, xq)
        head0 = pilot_fine_tuning.fit_head(cls_feats(base, center(su)), ys, np.arange(len(ys)), None)
        with torch.no_grad():
            acc["lp"][e] = (head0(torch.from_numpy(Xq)).argmax(1).numpy() == yq).mean()
        seed = 1_000_000 * shot + e + (500_000 if ep else 0)
        for lab, m, cfg in plan:
            net = fine_tuning.Net(B, copy.deepcopy(base), ["cls"], copy.deepcopy(head0)).to(dev)
            params = method_params(net, m, cfg, seed)
            torch.manual_seed(seed)                    # the crops: the same for every start and configuration
            try:
                adapt(net, params, lambda: (augment_all(su, dev), ysd), cfg["steps"], cfg["lr"], dev)
                acc[lab][e] = (predict(net, xq) == yq).mean()
            except FloatingPointError:
                acc[lab][e], div[lab][e] = 1 / WAYS, True
            del net, params
        if (e + 1) % 50 == 0 or e + 1 == n:
            print(f"{c} {name} {shot}-shot: {e + 1}/{n} episodes in {time.time() - t0:.0f}s; mean so far "
                  + ", ".join(f"{k} {100 * np.nanmean(v):.1f}" for k, v in acc.items()), flush=True)
    classifiers.save(dst, {**{f"acc|{k}": v for k, v in acc.items()}, **{f"div|{k}": v for k, v in div.items()}})


def dev_a(task):
    name, shot, m = DEV_A_TASKS[task]
    t0 = time.time()
    plan = [(cfg_name(m, cfg), m, cfg) for cfg in GRIDS[m]]
    episode_run("imagenet", name, shot, plan, OUT / "dev_a" / f"imagenet__{name}__{shot}__{m}.npz")
    manifest("dev_a", task, network="imagenet", dataset=name, shot=shot, method=m,
             episodes_sha256=sha256(DATA_OUT / f"{name}_episodes.npz"), seconds=round(time.time() - t0))


def gonogo(task):
    """The go/no-go check for one network on one go/no-go domain's own episodes. It runs full fine-tuning at plain's
    choice for the domain's group, and the linear probe that every run gives. The manifest keeps dev.json's hash."""
    c, name = GNG_TASKS[task]
    t0 = time.time()
    ch = json.load(open(OUT / "dev.json"))["choice"][GROUP_OF[name]][str(GNG_SHOT)]["ft"]
    episode_run(c, name, GNG_SHOT, [("ft", "ft", ch)], OUT / "gonogo" / f"{c}__{name}.npz", ep="g")
    manifest("gonogo", task, network=c, dataset=name, chosen=ch, dev_sha256=sha256(OUT / "dev.json"),
             episodes_sha256=sha256(DATA_OUT / f"{name}_episodes.npz"), seconds=round(time.time() - t0))


def final_a(task):
    """The final grid for one network, target and shot. It runs the linear probe, LoRA and full fine-tuning at plain's
    choices for the target's domain group. The manifest keeps dev.json's hash."""
    c, name, shot = FINAL_A_TASKS[task]
    t0 = time.time()
    ch = json.load(open(OUT / "dev.json"))["choice"][GROUP_OF[name]][str(shot)]
    plan = [(m, m, ch[m]) for m in ("lora", "ft")]
    episode_run(c, name, shot, plan, OUT / "final_a" / f"{c}__{name}__{shot}.npz")
    manifest("final_a", task, network=c, dataset=name, shot=shot, chosen=ch, dev_sha256=sha256(OUT / "dev.json"),
             episodes_sha256=sha256(DATA_OUT / f"{name}_episodes.npz"), seconds=round(time.time() - t0))


def step0():
    """Smoke-test checks on one ISIC 5-shot episode from plain.
      - At step 0, full fine-tuning (each block choice) and LoRA reproduce the linear probe's predictions on at least
        99% of the queries. The probe is fit on float16-autocast features.
      - A few steps of LoRA move the adapters and the head, and nothing else.
      - A few steps of full fine-tuning move the chosen blocks, the final norm and the head, and more only when every
        block trains.
      - No PAD-UFES-20 episode shows a lesion twice."""
    dev = pilot_fine_tuning.gpu_setup()
    E, pos, cache = episode_files("isic")
    shot, e = 5, 0
    su = torch.from_numpy(np.stack([cache[pos[i]] for i in E[f"sup{shot}"][e].ravel()])).to(dev)
    xq = center(torch.from_numpy(np.stack([cache[pos[i]] for i in E[f"qry{shot}"][e].ravel()])).to(dev))
    ys = np.repeat(np.arange(WAYS), shot)
    base = screen.build(B, "imagenet").to(dev).eval()
    nb = len(base.blocks)
    head0 = pilot_fine_tuning.fit_head(cls_feats(base, center(su)), ys, np.arange(len(ys)), None)
    with torch.no_grad():
        lp = head0(torch.from_numpy(cls_feats(base, xq))).argmax(1).numpy()
    out = {}
    for m, cfg in [("ft", dict(lr=1e-2, steps=0, blocks=b)) for b in BLOCKS] + [("lora", dict(rank=4, lr=1e-2, steps=0))]:
        net = fine_tuning.Net(B, copy.deepcopy(base), ["cls"], copy.deepcopy(head0)).to(dev)
        method_params(net, m, cfg, 0)
        lab = cfg_name(m, cfg)
        out[lab] = float((predict(net, xq) == lp).mean())
        print(f"{lab} at step 0 agrees with the linear probe on {100 * out[lab]:.1f}% of the queries", flush=True)
        assert out[lab] >= 0.99, f"{lab} at step 0 does not reproduce the linear probe"
    ysd = torch.as_tensor(ys, device=dev)
    for m, cfg in [("lora", dict(rank=4, lr=1e-2, steps=5))] + [("ft", dict(lr=1e-2, steps=5, blocks=b)) for b in BLOCKS]:
        net = fine_tuning.Net(B, copy.deepcopy(base), ["cls"], copy.deepcopy(head0)).to(dev)
        params = method_params(net, m, cfg, 0)
        before = {n: v.detach().clone() for n, v in net.state_dict().items()}
        torch.manual_seed(0)
        adapt(net, params, lambda: (augment_all(su, dev), ysd), cfg["steps"], cfg["lr"], dev)
        changed = {n for n, v in net.state_dict().items() if not torch.equal(before[n], v)}
        if m == "lora":
            adapters = {n for n in before if ".parametrizations." in n and not n.endswith(".original")}
            assert changed == adapters | {"head.fc.weight", "head.fc.bias"}, sorted(changed ^ adapters)
            assert len(adapters) == 4 * nb
        else:
            blocks = {int(n.split(".")[2]) for n in changed if n.startswith("backbone.blocks.")}
            assert blocks == set(range(nb - cfg["blocks"], nb)), sorted(blocks)
            rest = {n for n in changed if not n.startswith(("backbone.blocks.", "backbone.norm.", "head."))}
            assert cfg["blocks"] == nb or not rest, sorted(rest)
        lab = cfg_name(m, cfg)
        out[f"{lab}_changed"] = len(changed)
        print(f"{lab}: {len(changed)} tensors moved in {cfg['steps']} steps, as expected", flush=True)
    n_rep = repeated_groups(episode_files("pad")[0], pd.read_csv(DATA_OUT / "pad_index.csv")["group"].values)
    print(f"PAD-UFES-20: {n_rep} episodes show a lesion twice", flush=True)
    assert n_rep == 0
    manifest("step0", agreement=out)


# ---------------------------------------------------------------- development choices
def choose():
    """Plain's choices -> dev.json and dev_choices.csv.

    For each domain group, shot and method, the choice is the grid configuration with the highest mean development
    accuracy over the group's domains. The skin group is PAD-UFES-20, for ISIC. The general group is CUB and Places,
    for ChestX, EuroSAT and CropDisease. Ties go to the earlier configuration in the grid. Ours runs with plain's
    choices. dev.json also keeps every configuration's accuracy per domain, and plain's PAD-UFES-20 5-shot accuracy for
    each method at its choice. The headline method (HEADLINE) is chosen from these."""
    acc = {}                                           # (domain, shot, label) -> mean accuracy (%)
    for name, shot, m in DEV_A_TASKS:
        z = np.load(OUT / "dev_a" / f"imagenet__{name}__{shot}__{m}.npz")
        for lab in ["lp"] + [cfg_name(m, cfg) for cfg in GRIDS[m]]:
            acc[name, shot, lab] = 100 * float(z[f"acc|{lab}"].mean())
    res, rows = {"choice": {}, "groups": GROUPS}, []
    for g, doms in GROUPS.items():
        for shot in SHOTS:
            mean = lambda lab: float(np.mean([acc[n, shot, lab] for n in doms]))
            ch = {m: max(GRIDS[m], key=lambda cfg: mean(cfg_name(m, cfg))) for m in GRIDS}
            res["choice"].setdefault(g, {})[str(shot)] = ch
            rows += [dict(group=g, shot=shot, method="lp", setting="C=0.1", chosen_on="+".join(doms), dev_acc=mean("lp"))]
            rows += [dict(group=g, shot=shot, method=m, setting=cfg_name(m, cfg), chosen_on="+".join(doms),
                          dev_acc=mean(cfg_name(m, cfg))) for m, cfg in ch.items()]
    skin = res["choice"]["skin"][str(GNG_SHOT)]
    res["pad_5shot"] = {"lp": acc["pad", GNG_SHOT, "lp"], **{m: acc["pad", GNG_SHOT, cfg_name(m, skin[m])] for m in GRIDS}}
    res["per_domain"] = {f"{n}|{s}|{lab}": v for (n, s, lab), v in acc.items()}
    json.dump(res, open(OUT / "dev.json", "w"), indent=1)
    pd.DataFrame(rows).to_csv(OUT / "dev_choices.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"), flush=True)
    print("plain's PAD-UFES-20 5-shot accuracy at each method's choice:", json.dumps(res["pad_5shot"]), flush=True)
    manifest("choose", dev_sha256=sha256(OUT / "dev.json"), choice=res["choice"])


def gonogo_summary():
    """gonogo.csv and gonogo.txt. For each go/no-go domain, under full fine-tuning (the check) and the linear probe
    (for reference), they give plain's and ours' accuracy, and ours - plain paired on episodes, with the benchmark's
    intervals. gonogo.txt also lists the measured time of every development and go/no-go job."""
    ours = ["imagenet"] if QUICK else [f"{FAM}_s{s}" for s in SEEDS]   # the smoke test: plain stands in for ours
    dv = json.load(open(OUT / "dev.json"))
    rows = []
    for name in GNG_DOMAINS:
        zp = np.load(OUT / "gonogo" / f"imagenet__{name}.npz")
        zo = [np.load(OUT / "gonogo" / f"{c}__{name}.npz") for c in ours]
        ch = dv["choice"][GROUP_OF[name]][str(GNG_SHOT)]["ft"]
        for m in ("ft", "lp"):
            a_p, a_o = 100 * zp[f"acc|{m}"][None], 100 * np.stack([z[f"acc|{m}"] for z in zo])
            mp, plo, phi, _ = interval_a(a_p)
            mo, olo, ohi, oi = interval_a(a_o)
            g, glo, ghi, gi = interval_a(a_o - a_p)
            rows.append(dict(domain=name, method=m, setting=cfg_name(m, ch) if m == "ft" else "C=0.1",
                             episodes=a_p.shape[1], seeds=len(ours), plain=mp, plain_lo=plo, plain_hi=phi, ours=mo,
                             ours_lo=olo, ours_hi=ohi, ours_interval=oi, gain=g, gain_lo=glo, gain_hi=ghi,
                             gain_interval=gi, diverged_plain=int(zp[f"div|{m}"].sum()),
                             diverged_ours=int(sum(z[f"div|{m}"].sum() for z in zo))))
    G = pd.DataFrame(rows)
    G.to_csv(OUT / "gonogo.csv", index=False)
    jobs = [json.load(open(f)) for f in sorted((OUT / "manifests").glob("*.json"))]
    T = (pd.DataFrame([dict(step=j["step"], gpu=j["gpu"], hours=j["seconds"] / 3600) for j in jobs
                       if j["step"] in ("dev_a", "gonogo")])
         .groupby(["step", "gpu"])["hours"].agg(["count", "sum", "mean"]).reset_index())
    fmt = lambda df: df.to_string(index=False, float_format=lambda v: f"{v:.2f}")
    txt = ["Leaves to Lesions -- the go/no-go check. DINO ViT-S/16; plain = ImageNet "
           f"DINO; ours = synthetic pairs on COIL-100 views, seeds {SEEDS}. 5-way {GNG_SHOT}-shot, 15 queries, "
           f"{N_GNG} episodes per domain; full fine-tuning at plain's development choice; accuracy %.", "",
           fmt(G), "", f"dev.json sha256 {sha256(OUT / 'dev.json')}", "",
           "Measured GPU-hours (count of jobs, total, mean per job):", fmt(T)]
    (OUT / "gonogo.txt").write_text("\n".join(txt) + "\n")
    print("\n".join(txt), flush=True)
    manifest("gonogo_summary", rows=rows)


# ---------------------------------------------------------------- summary
def boot_mean(D, n_boot=None):
    """Hierarchical bootstrap of the mean of D (seeds x episodes). Seeds and episodes are resampled with replacement,
    independently (RandomState(0), corrected_pilot.N_BOOT resamples). The seed effect is inflated by sqrt(n / (n - 1)),
    as in screen.Boot."""
    rng, n_boot = np.random.RandomState(0), n_boot or corrected_pilot.N_BOOT
    ns, ne = D.shape
    Ce = np.stack([np.bincount(r, minlength=ne) for r in rng.randint(0, ne, (n_boot, ne))]).astype(float)
    Cs = np.stack([np.bincount(r, minlength=ns) for r in rng.randint(0, ns, (n_boot, ns))]).astype(float)
    S = Ce @ D.T / ne                                  # each seed's mean over the resampled episodes
    if ns > 1:
        mu = S.mean(1, keepdims=True)
        S = mu + np.sqrt(ns / (ns - 1)) * (S - mu)
    return (Cs * S).sum(1) / ns


def interval_a(D):
    """Mean of D (seeds x episodes, %) and its 95% interval, as the plan fixes it. For one network, the interval is
    1.96 SD / sqrt(episodes). For several seeds, it is the wider of the hierarchical bootstrap's percentile interval
    and the t-interval over the seed means."""
    mean = float(D.mean())
    if D.shape[0] == 1:
        h = 1.96 * D[0].std(ddof=1) / np.sqrt(D.shape[1])
        return mean, mean - h, mean + h, "episodes"
    lo, hi = np.percentile(boot_mean(D), [2.5, 97.5])
    _, tlo, thi, _ = lesions.t_interval(D.mean(1))
    return (mean, tlo, thi, "t") if thi - tlo > hi - lo else (mean, float(lo), float(hi), "bootstrap")


def summarize_a(ours):
    """benchmark_results.csv's rows. For each target, shot and method, they give the setting and the development domains
    it was chosen on, plain's and ours' accuracy, and ours - plain paired on episodes, with the benchmark's
    intervals."""
    dv = json.load(open(OUT / "dev.json"))
    rows = []
    for name in DATA_A:
        for shot in SHOTS:
            fp = OUT / "final_a" / f"imagenet__{name}__{shot}.npz"
            fo = [OUT / "final_a" / f"{c}__{name}__{shot}.npz" for c in ours]
            if not (fp.exists() and all(f.exists() for f in fo)):
                continue
            zp, zo = np.load(fp), [np.load(f) for f in fo]
            ch = dv["choice"][GROUP_OF[name]][str(shot)]
            for m in METHODS:
                a_p = 100 * zp[f"acc|{m}"][None]
                a_o = 100 * np.stack([z[f"acc|{m}"] for z in zo])
                mp, plo, phi, pi = interval_a(a_p)
                mo, olo, ohi, oi = interval_a(a_o)
                g, glo, ghi, gi = interval_a(a_o - a_p)
                rows.append(dict(target=name, shot=shot, method=m, setting=cfg_name(m, ch[m]) if m in ch else "C=0.1",
                                 chosen_on="+".join(dv["groups"][GROUP_OF[name]]) if m in ch else "",
                                 episodes=a_p.shape[1], seeds=len(ours), plain=mp, plain_lo=plo, plain_hi=phi,
                                 plain_interval=pi, ours=mo, ours_lo=olo, ours_hi=ohi, ours_interval=oi, gain=g,
                                 gain_lo=glo, gain_hi=ghi, gain_interval=gi, diverged_plain=int(zp[f"div|{m}"].sum()),
                                 diverged_ours=int(sum(z[f"div|{m}"].sum() for z in zo))))
    return pd.DataFrame(rows)


def leaderboard_rows():
    """docs/leaderboard.md's 5-way 5- and 1-shot tables, one row per published entry, with its comparability mark and
    its four accuracies. A printed "± h" is dropped, and "—" (not reported) becomes missing."""
    rows, shot, head = [], None, None
    for line in LEADERBOARD.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("## "):
            shot, head = {"## 5-way 5-shot": 5, "## 5-way 1-shot": 1}.get(line.strip()), None
        elif shot and line.startswith("| Method "):
            head = cells
        elif shot and head and line.startswith("| ") and len(cells) == len(head):
            r = dict(zip(head, cells))
            num = lambda s: np.nan if s == "—" else float(s.replace("*", "").split("±")[0])
            rows.append(dict(shot=shot, method=r["Method"], paper=r["Paper"], ft=r["FT"], comparable=r["Comparable"],
                             **{LB_NAMES[k]: num(r[k]) for k in LB_NAMES}))
    return pd.DataFrame(rows)


def leaderboard(A):
    """leaderboard.csv. For each target and shot, plain and ours with the headline method are set against the best
    published row that is directly comparable ("Yes" in docs/leaderboard.md), and the best that is at least probably
    comparable. Gaps are signed, so losses show."""
    L = leaderboard_rows()
    rows = []
    for name in DATA_A:
        for shot in SHOTS:
            a = A[(A.target == name) & (A.shot == shot) & (A.method == HEADLINE)]
            if not len(a):
                continue
            a = a.iloc[0]
            row = dict(target=name, shot=shot, method=HEADLINE, plain=a.plain, ours=a.ours, ours_lo=a.ours_lo,
                       ours_hi=a.ours_hi)
            for kind, marks in (("direct", ("Yes",)), ("probable", ("Yes", "Probably"))):
                cand = L[(L.shot == shot) & L.comparable.map(lambda s: s.startswith(marks))].dropna(subset=[name])
                b = cand.loc[cand[name].idxmax()]
                row.update({f"best_{kind}": b[name], f"best_{kind}_row": f"{b.method}, {b.paper}, FT {b.ft}",
                            f"plain_minus_best_{kind}": a.plain - b[name], f"ours_minus_best_{kind}": a.ours - b[name]})
            rows.append(row)
    return pd.DataFrame(rows)


def summarize():
    """The final grid's tables and the plan's verdicts: benchmark_results.csv (every cell), leaderboard.csv,
    verdict.json and summary.txt. summary.txt also shows REAP's source stage and frozen test from reap/reap.csv, once
    the REAP script has written it."""
    ours = ["imagenet", "imagenet"] if QUICK else [f"{FAM}_s{s}" for s in SEEDS]   # the smoke test: plain as both
    A = summarize_a(ours)
    A.to_csv(OUT / "benchmark_results.csv", index=False)
    Lb = leaderboard(A)
    Lb.to_csv(OUT / "leaderboard.csv", index=False)
    verdict = {}
    prim = A[(A.target == "isic") & (A.shot == 5) & (A.method == "ft")]
    if len(prim):
        r = prim.iloc[0]
        verdict["primary"] = dict(claim="ours beats plain on ISIC 5-shot under full fine-tuning (95% interval above 0)",
                                  gain=r.gain, lo=r.gain_lo, hi=r.gain_hi, holds=bool(r.gain_lo > 0))
    head = A[(A.target == "isic") & (A.shot == 5) & (A.method == HEADLINE)]
    if len(head):
        r = head.iloc[0]
        h = (r.ours_hi - r.ours_lo) / 2
        bar = math.sqrt(h ** 2 + REAP_H ** 2)
        verdict["headline"] = dict(claim=f"ours with {HEADLINE} beats REAP's {REAP} on ISIC 5-way 5-shot by more than "
                                         f"the combined 95% half-width", ours=r.ours, h_ours=h, needed=REAP + bar,
                                   margin=r.ours - REAP, holds=bool(r.ours - REAP > bar))
    one = A[(A.target == "isic") & (A.shot == 1) & (A.method == HEADLINE)]
    if len(one):
        r = one.iloc[0]
        verdict["isic_1shot"] = dict(note=f"descriptive, not a claim: plain and ours with {HEADLINE} against REAP's "
                                          f"{REAP_1}", plain=r.plain, ours=r.ours, ours_lo=r.ours_lo, ours_hi=r.ours_hi)
    json.dump(verdict, open(OUT / "verdict.json", "w"), indent=1, default=float)
    reap = OUT / "reap" / "reap.csv"
    fmt = lambda df: df.to_string(index=False, float_format=lambda v: f"{v:.2f}")
    txt = ["Leaves to Lesions -- the final grid, run once. "
           f"DINO ViT-S/16; plain = ImageNet DINO; ours = synthetic pairs on COIL-100 views, seeds {SEEDS}; the "
           f"headline's method: {HEADLINE}.",
           "", "Development choices (dev_choices.csv; on plain, never on a target):",
           fmt(pd.read_csv(OUT / "dev_choices.csv")),
           "", "The benchmark episodes (BSCD-FSL, 5-way, 15 queries; accuracy %; ours - plain paired on episodes):", fmt(A),
           "", f"Against docs/leaderboard.md ({HEADLINE}; gaps signed, losses included):", fmt(Lb),
           "", "REAP's source stage from plain and from ours, then its frozen test (reap/reap.csv):",
           fmt(pd.read_csv(reap)) if reap.exists() else "(not run yet)",
           "", "Verdicts:", json.dumps(verdict, indent=1, default=float)]
    (OUT / "summary.txt").write_text("\n".join(txt) + "\n")
    print("\n".join(txt), flush=True)
    manifest("summarize", verdict=verdict)


STEPS_TASKED = {"make", "train", "features", "evaluate", "data", "dev_a", "gonogo", "final_a"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=sorted(STEPS_TASKED | {"check", "crop_check", "step0", "choose", "gonogo_summary",
                                                           "summarize"}))
    ap.add_argument("--task", type=int)
    a = ap.parse_args()
    fn = globals()[a.step]
    if a.step in STEPS_TASKED:
        fn(a.task)
    else:
        fn()


if __name__ == "__main__":
    main()
