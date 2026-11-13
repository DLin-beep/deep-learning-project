# Leaves to Lesions

The main code behind our paper *Leaves to Lesions: Does Sharing Visual Properties Predict Transfer?* by Derek Lin,
Dhruv Dudhat, Qihao Bu and Suabcua Lee.

## Objective

More often than not, medical images are scarcer and costlier than images from more accessible domains, such as the
leaf photos in PlantVillage. Not only are images from these domains more abundant, but they also rarely need IRB
review, consent forms, or a data use agreement. Our objective is to find out whether training a network on one of
these accessible domains first helps it learn a medical task from only a few labels. We also want to know whether the
visual properties that the two domains share can tell us in advance when it will. We focus on skin lesions and leaf
spots, since they share many visual cues, such as dark specks, ragged borders and rings.

## What we found

Shared visual properties do not predict transfer. Across four sources, four ImageNet-pretrained networks and three
medical targets, the distance between the datasets' visual properties predicts none of the gains, and almost no source
helps. One gain does hold, from plant disease to skin lesions. It comes from the task of telling healthy from diseased
on a whole object, since species labels on the same leaves give no gain. Synthetic lesions pasted onto healthy leaves
and everyday objects work as well as real disease. Most of the gain comes from reweighting features the network
already has, and it is narrow. It shows up only with few labels, on skin, and in self-supervised and CLIP-style
transformers. Five claims that we fixed in advance all hold on a new skin dataset, and the paper reports the numbers
behind each finding.

## What is in this repository

We kept only the main files, so that the method is easy to follow. The scripts are the main ones from our full
codebase. They rely on smaller helper modules and cluster job scripts that are not included here, so they are for
reading rather than running as they are.

| File | What it does |
|---|---|
| `METHOD.md` | Our method, step by step |
| `DATA.md` | Every dataset and network that we used, where it comes from and its license |
| `figures/` | The paper's four figures |
| `code/e1_screen.py` | Trains each network on a source, tests it few-shot on each target, and measures the attribute distance between datasets |
| `code/e15_pointing.py` | Makes the synthetic lesions and trains a network to score each lesioned image above its clean copy |
| `code/e16_adaptation.py` | Trains the synthetic-lesion network on COIL-100 and adapts it to the BSCD-FSL benchmark with a linear probe, LoRA or full fine-tuning |
| `code/e9_linear_stand_in.py` | Tests whether plant training only reweights the features that the network already has, with a linear map fit without labels |
| `code/e2_confirmatory_bcn20000.py` | Tests the five claims that we fixed in advance on BCN20000, a skin dataset that we had not used before |
| `code/common/` | Shared helpers for the networks and their readouts, the datasets, the few-shot classifiers and their evaluation, the hierarchical bootstrap, the linear map and the benchmark's episodes |
| `requirements.txt` | The package versions that we used |

## Data

None of the datasets or network weights are in this repository, since each one comes with its own terms, and several
of them allow only non-commercial use. `DATA.md` lists where each one comes from.

## Citation

```bibtex
@misc{lin2026leaves,
  title  = {Leaves to Lesions: Does Sharing Visual Properties Predict Transfer?},
  author = {Lin, Derek and Dudhat, Dhruv and Bu, Qihao and Lee, Suabcua},
  year   = {2026},
  note   = {Preprint}
}
```
