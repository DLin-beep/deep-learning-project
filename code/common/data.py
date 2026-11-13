"""Images and targets.

A target's table has the columns path, y and group. The group is the unit that never crosses a split: the lesion, the
patient or the image. Its cache holds the images resized (shorter side) and centre-cropped to 224 px, as uint8
N x 3 x 224 x 224 in the table's order."""
import io
import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from common.paths import DATA, SCREEN_OUT, CREDIBILITY_OUT, ROOT

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
EVAL_TF = transforms.Compose([
    transforms.Resize(224), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
HAM_CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
TARGETS = ["ham", "ddr", "cxr"]                 # the screen's targets: cached and featurized in SCREEN_OUT
CXR_CLASSES = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumothorax"]
CXR_MAX = 12000
ISIC = DATA / "isic2018"
ATTRS = ["globules", "milia_like_cyst", "negative_network", "pigment_network", "streaks"]


# ---------------------------------------------------------------- the machine
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def cpus():
    return int(os.environ.get("SLURM_CPUS_PER_TASK", "6"))


class ImageList(Dataset):
    """Items are (path_or_bytes, label)."""
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        src, y = self.items[i]
        img = Image.open(io.BytesIO(src) if isinstance(src, bytes) else src).convert("RGB")
        return self.tf(img), y


def cache_images(items, dst, batch_size, size=224, workers=None):
    """Resize and centre-crop items (path or bytes, label) to size px and write them in order to dst as uint8
    N x 3 x size x size, through a .tmp file. `workers` loader processes read them (default min(8, cpus())).
    Returns the number of images."""
    tf = transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size), transforms.PILToTensor()])
    dl = torch.utils.data.DataLoader(ImageList(items, tf), batch_size=batch_size,
                                     num_workers=min(8, cpus()) if workers is None else workers)
    tmp = dst.with_suffix(".tmp.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(items), 3, size, size))
    i = 0
    for x, _ in dl:
        arr[i:i + len(x)] = x.numpy(); i += len(x)
    assert i == len(items)
    arr.flush(); del arr
    os.replace(tmp, dst)
    return i


# ---------------------------------------------------------------- the screen's targets
def ham_table():
    gt = pd.read_csv(DATA / "ham" / "ISIC2018_Task3_Training_GroundTruth" / "ISIC2018_Task3_Training_GroundTruth.csv")
    gt["y"] = gt[HAM_CLASSES].values.argmax(1)
    grp = pd.read_csv(DATA / "ham" / "lesion_groupings.csv")
    df = gt.merge(grp[["image", "lesion_id"]], on="image", how="left")
    assert df["lesion_id"].notna().all()
    df["path"] = [str(DATA / "ham" / "ISIC2018_Task3_Training_Input" / f"{i}.jpg") for i in df["image"]]
    return df


def n_classes(t):
    return {"ham": 7, "ddr": 5, "cxr": 7}[t]


def target_table(t):
    """Columns: path, y, group (the unit that never crosses a split)."""
    if t == "ham":
        df = ham_table()
        return pd.DataFrame(dict(path=df["path"], y=df["y"], group=df["lesion_id"]))
    if t == "ddr":
        rows = []
        for split in ["train", "valid", "test"]:
            for line in open(DATA / "ddr" / "DR_grading" / f"{split}.txt"):
                name, grade = line.split()
                if int(grade) <= 4:                       # grade 5 = ungradable
                    rows.append((str(DATA / "ddr" / "DR_grading" / split / name), int(grade), name))
        return pd.DataFrame(rows, columns=["path", "y", "group"])
    if t == "cxr":
        d = pd.read_csv(DATA / "cxr" / "Data_Entry_2017_v2020.csv")
        d = d[d["Finding Labels"].isin(CXR_CLASSES)]
        d = d[[(DATA / "cxr" / "images" / f).exists() for f in d["Image Index"]]]
        patients = d["Patient ID"].unique()
        np.random.RandomState(0).shuffle(patients)
        size = d.groupby("Patient ID").size()
        keep = patients[: int(np.searchsorted(np.cumsum(size.loc[patients].values), CXR_MAX)) + 1]
        d = d[d["Patient ID"].isin(set(keep))].sort_values("Image Index")
        return pd.DataFrame(dict(path=[str(DATA / "cxr" / "images" / f) for f in d["Image Index"]],
                                 y=[CXR_CLASSES.index(f) for f in d["Finding Labels"]],
                                 group=d["Patient ID"].values))
    raise ValueError(t)


def cache(t, out=SCREEN_OUT):
    """Write a screen target's index and cache to out/cache. HAM10000 reuses the pilot fine-tuning's cache if there."""
    d = out / "cache"
    d.mkdir(parents=True, exist_ok=True)
    tab = target_table(t).reset_index(drop=True)
    tab.to_csv(d / f"{t}_index.csv", index=False)
    dst = d / f"{t}_u8.npy"
    old = ROOT / "outputs" / "pilot_fine_tuning" / "eval_u8.npy"           # same preprocessing, same image order
    if t == "ham" and old.exists() and np.load(old, mmap_mode="r").shape[0] == len(tab):
        if not dst.exists():
            os.symlink(old, dst)
        print("ham: reusing", old); return
    i = cache_images(list(zip(tab["path"], tab["y"])), dst, 256)
    print(f"{t}: cached {i} images; class counts {np.bincount(tab['y'], minlength=n_classes(t)).tolist()}; "
          f"{tab['group'].nunique()} groups")


def load_target(t, out=SCREEN_OUT):
    tab = pd.read_csv(out / "cache" / f"{t}_index.csv")
    return tab, np.load(out / "cache" / f"{t}_u8.npy", mmap_mode="r")


# ---------------------------------------------------------------- ISIC 2018 and the later targets
def isic_table(t):
    if t == "isic_test":
        gt = pd.read_csv(ISIC / "ISIC2018_Task3_Test_GroundTruth" / "ISIC2018_Task3_Test_GroundTruth.csv")
        return pd.DataFrame(dict(path=[str(ISIC / "ISIC2018_Task3_Test_Input" / f"{i}.jpg") for i in gt["image"]],
                                 y=gt[HAM_CLASSES].values.argmax(1), group=gt["image"]))
    if t == "isic_attr":
        d = ISIC / "ISIC2018_Task1-2_Training_Input"
        m = ISIC / "ISIC2018_Task2_Training_GroundTruth_v3"
        ids = sorted(p.stem for p in d.glob("ISIC_*.jpg"))
        rows = []
        for i in ids:
            lab = [int(np.asarray(Image.open(m / f"{i}_attribute_{a}.png")).max() > 0) for a in ATTRS]
            rows.append(dict(path=str(d / f"{i}.jpg"), y=0, group=i, **dict(zip(ATTRS, lab))))
        return pd.DataFrame(rows)
    raise ValueError(t)


def cache_isic(t, out=CREDIBILITY_OUT):
    d = out / "cache"
    d.mkdir(parents=True, exist_ok=True)
    tab = isic_table(t)
    tab.to_csv(d / f"{t}_index.csv", index=False)
    i = cache_images(list(zip(tab["path"], tab["y"])), d / f"{t}_u8.npy", 128)
    extra = {a: int(tab[a].sum()) for a in ATTRS} if t == "isic_attr" else np.bincount(tab["y"], minlength=7).tolist()
    print(f"{t}: cached {i} images; {extra}")


def load_isic(t, out=CREDIBILITY_OUT):
    """Any target cached outside the screen's folder (the name is historical)."""
    return pd.read_csv(out / "cache" / f"{t}_index.csv"), np.load(out / "cache" / f"{t}_u8.npy", mmap_mode="r")


def target_dir(b, t, screen_out=SCREEN_OUT, credibility_out=CREDIBILITY_OUT):
    """Where a backbone's features and predictions on a target are kept."""
    return (screen_out / b / t) if t in TARGETS else (credibility_out / b / t)
