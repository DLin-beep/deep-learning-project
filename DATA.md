# Data

No datasets or network weights are in this repository, since each dataset comes with its own terms and most of them
are large. This file lists every dataset and network that we used, where it comes from, the folder under `data/` where
the code looks for it, and its license where we recorded one. Where the tables say "see source", please check the
terms at the source before you use or share the images, since several of these datasets allow only non-commercial use.

## Sources

| Dataset | Folder | Source | License |
|---|---|---|---|
| PlantVillage, segmented, 38 classes | `data/plantvillage/` | GitHub `spMohanty/PlantVillage-Dataset` | see source |
| Food-101 | `data/food101/` | Hugging Face `ethz/food101` | see source |
| DTD | `data/dtd/` | Oxford VGG | see source |
| NCT-CRC-HE-100K | `data/nct/` | Zenodo record 1214456 | see source |
| COIL-100 | `data/coil100/coil-100/` | Columbia's CAVE lab | see source |
| VisA | `data/visa/` | Amazon S3, `amazon-visual-anomaly` | CC BY 4.0 |
| FruitNet | `data/fruitnet/` | Mendeley Data `b6fftwbr2v`, version 3 | CC BY 4.0 |
| MatsyaDx-BD fish photos | `data/matsyadx/` | Mendeley Data `sxkynv9t7n`, version 2 | CC BY 4.0 |
| Leaf images of Chouhan et al. | `data/chouhan/` | Mendeley Data `hb74ynkjcn`, version 5 | CC BY 4.0 |
| miniImageNet, 64 training classes of ILSVRC 2012 | `data/imagenet/` | Kaggle `imagenet-object-localization-challenge`, with the split from GitHub `twitter/meta-learning-lstm` | ImageNet's non-commercial terms |

## Targets

| Dataset | Folder | Source | License |
|---|---|---|---|
| HAM10000, the ISIC 2018 Task 3 training set | `data/ham/` | The ISIC 2018 challenge's download site | CC BY-NC 4.0 |
| ISIC 2018, the Task 3 test set and the Task 1 and 2 images and masks | `data/isic2018/` | The ISIC 2018 challenge's download site | CC BY-NC |
| HAM10000 lesion masks of Tschandl et al. | `data/ham10000_seg/` | Harvard Dataverse, file 3838943 | see source |
| DDR, diabetic retinopathy grades | `data/ddr/` | Hugging Face mirror `ctmedtech/DDR-dataset` | CC BY 4.0 |
| ChestX-ray14 | `data/cxr/` | Hugging Face mirror `alkzar90/NIH-Chest-X-ray-dataset` | see source |
| BCN20000 | `data/bcn20000/` | ISIC Archive collection 249 | CC BY-NC |
| PAD-UFES-20 | `data/pad_ufes20/` | Mendeley Data `zr7vgbcyr2` | CC BY 4.0 |
| HyperKvasir, 6 classes | `data/hyperkvasir6/` | OSF `mh9sj` | CC BY 4.0 |
| BUSI breast ultrasound, curated | `data/busi_curated/` | Zenodo record 19047974 | CC BY 4.0 |
| HIBA | `data/hiba_derm/` | ISIC Archive collection 175 | CC BY |
| HAM10000 body sites | `data/ham_sites/` | ISIC Archive collection 212 | see source |
| MVTec AD | `data/mvtec_ad/` | Hugging Face mirror `foersben/mvtec-ad` | CC BY-NC-SA 4.0 |
| Oxford-IIIT Pets | `data/oxford_pets/` | Oxford VGG | CC BY-SA 4.0 |
| PlantDoc | `data/plantdoc/` | GitHub `pratikkayal/PlantDoc-Dataset` | CC BY 4.0 |
| EuroSAT, RGB | `data/eurosat/` | Zenodo record 7711810 | see source |
| CropDisease, BSCD-FSL's copy of PlantVillage | `data/cropdisease/` | Kaggle `saroz014/plant-disease` | see source |
| CUB-200-2011, for development only | `data/cub/` | CaltechDATA | see source |
| Places365 validation set, for development only | `data/places365/` | MIT CSAIL | see source |

PAD-UFES-20 is a target in the paper's experiments, but on the benchmark we only used it for development. "ChestX"
also means two different sets of images, since the screen uses a cache that is capped by patient at 12,000 images,
while the benchmark uses all 25,848 single-label images.

## Networks

| Network | Source | License |
|---|---|---|
| Supervised ResNet-50 | torchvision `IMAGENET1K_V1` | see source |
| DINO ResNet-50, ViT-S/16 and ViT-B/16 | Meta's DINO release | Apache 2.0 |
| Supervised ViT-S/16 | timm `deit_small_patch16_224` | see source |
| CLIP ViT-B/16 and ViT-L/14 | timm `vit_base_patch16_clip_224.openai` and `vit_large_patch14_clip_quickgelu_224.openai` | MIT |
| DINOv2 ViT-S/14, ViT-B/14 and ViT-L/14 | timm `vit_small_patch14_dinov2.lvd142m` and its larger versions | Apache 2.0 |
| MONET | Hugging Face `suinleelab/monet` | CC BY-NC-SA 4.0 |
| DINOv3 ViT-L/16 | timm `vit_large_patch16_dinov3.lvd1689m` | DINOv3 License |
| PanDerm | Code from `github.com/SiyuanYan1/PanDerm`, with the checkpoint its authors release | CC BY-NC-ND 4.0 |

The DINOv3 License asks publications to acknowledge DINOv3. On the benchmark, we also compared against REAP, whose code
(`github.com/shuaiyi308/REAP` at commit `5ce30bc`) has no license, so it is not included here either.

## Good to know

CropDisease and the ImageNet images behind miniImageNet come from Kaggle, which needs an API token, and for ImageNet
you also need to accept the challenge's rules. COIL-100, the HAM10000 lesion masks and PanDerm have no download step in
our code, so we put them in place by hand. Several other folders, such as crops of the sources, COIL-100 copies with
painted blotches and the synthetic lesions, are built from these datasets by the preparation steps of our full
codebase. A later download may also not match ours, since PlantVillage and PlantDoc come from their default branch,
the Hugging Face downloads pin no revision, and DDR, ChestX-ray14 and MVTec AD come from unofficial mirrors.
