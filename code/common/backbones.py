"""Backbones and their readouts.

A condition is a network as released ("imagenet"), a source run's checkpoint loaded on top of it, or a norm-only
control (the release with only that run's normalization parameters). Readouts are the global-average-pooled stages
2-4 of a ResNet-50, and for a ViT the mean patch token after three blocks (before the final norm) plus the CLS token
after the final norm."""
import torch

from common.data import EVAL_TF, MEAN, STD, TRAIN_TF
from common.paths import DATA

# The stronger backbones run at their own training resolution; all others at 224 px.
# MICRO is their images per micro-batch in source training, where the default doesn't fit (other ViT-Ls use 8).
RES = {"vitb_dinov2_518": 518, "vitl_dinov2_518": 518, "vitl_dinov3_512": 512}
MICRO = {"vitb_dinov2_518": 8, "vitl_dinov2_518": 4, "vitl_dinov3_512": 4}
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
VIT_TRAINED = range(6, 12)                      # 0-indexed blocks a 12-block ViT trains on a source (the last 6)


def is_vit(b):
    return b.startswith(("vits", "vitb", "vitl", "clip"))


def stages(b):
    if b.startswith("vitl"):                            # 24 blocks: halfway, three quarters, last
        return ["b12", "b18", "b24"]
    return ["b6", "b9", "b12"] if is_vit(b) else ["s2", "s3", "s4"]


def readouts(b):
    return stages(b) + ["cat"] + (["cls"] if is_vit(b) else [])


def last(b):
    return stages(b)[-1]


def ctrl(b):
    return "ln" if is_vit(b) else "bn"


def head_keys(b):
    return {"head.weight", "head.bias"} if is_vit(b) else {"fc.weight", "fc.bias"}


def res(b):
    """Input size in pixels."""
    return RES.get(b, 224)


def micro_batch(b):
    """Images per micro-batch in source training (the same update, summed over them); None = the whole batch."""
    return MICRO.get(b, 8 if b.startswith("vitl") else None)


def transforms_at(b):
    """The paper's training and evaluation transforms (TRAIN_TF, EVAL_TF) at the backbone's resolution."""
    if res(b) == 224:
        return TRAIN_TF, EVAL_TF
    from torchvision import transforms
    r, norm = res(b), transforms.Normalize(MEAN, STD)
    return (transforms.Compose([transforms.RandomResizedCrop(r, scale=(0.5, 1.0)), transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(), norm]),
            transforms.Compose([transforms.Resize(r), transforms.CenterCrop(r), transforms.ToTensor(), norm]))


def resnet50_imagenet():
    from torchvision import models
    return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)


def imagenet_model(b):
    import timm
    from torchvision import models
    if b == "rn50_sup":
        return resnet50_imagenet()
    if b == "rn50_dino":
        m = models.resnet50(weights=None)
        sd = torch.load(DATA / "dino" / "dino_resnet50_pretrain.pth", map_location="cpu")
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not unexpected and set(missing) == head_keys(b), (missing, unexpected)
        return m
    if b == "vits_sup":
        return timm.create_model("deit_small_patch16_224", pretrained=True)
    if b == "clip_b16":
        return timm.create_model("vit_base_patch16_clip_224.openai", pretrained=True)
    if b == "vits_dinov2":                              # second-DINO check only; LVD-142M at 224 px
        return timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, img_size=224, num_classes=1000)
    if b in ("vitb_dinov2", "vitl_dinov2"):             # the stronger-model comparison; LVD-142M at 224 px
        arch = {"vitb_dinov2": "vit_base_patch14_dinov2.lvd142m", "vitl_dinov2": "vit_large_patch14_dinov2.lvd142m"}
        return timm.create_model(arch[b], pretrained=True, img_size=224, num_classes=1000)
    if b in ("vitb_dinov2_518", "vitl_dinov2_518"):     # stronger backbones at their training resolution
        arch = {"vitb_dinov2_518": "vit_base_patch14_dinov2.lvd142m",
                "vitl_dinov2_518": "vit_large_patch14_dinov2.lvd142m"}
        return timm.create_model(arch[b], pretrained=True, img_size=518, num_classes=1000)
    if b == "vitl_dinov3_512":                          # DINOv3 ViT-L/16 (LVD-1689M, DINOv3 License): RoPE, 4 registers
        return timm.create_model("vit_large_patch16_dinov3.lvd1689m", pretrained=True, num_classes=1000,
                                 global_pool="token")    # the head reads the CLS token, as upstream
    if b == "vitl_clip":                                # the stronger-model comparison: OpenAI CLIP ViT-L/14
        return timm.create_model("vit_large_patch14_clip_quickgelu_224.openai", pretrained=True, num_classes=1000)
    if b == "vitl_monet":                               # MONET: CLIP ViT-L/14 trained on dermatology literature
        m = timm.create_model("vit_large_patch14_clip_quickgelu_224", pretrained=False, num_classes=1000)
        missing, unexpected = m.load_state_dict(hf_clip_vision("suinleelab/monet"), strict=False)
        assert not unexpected and set(missing) == head_keys(b), (missing, unexpected)
        return m
    if b == "vitl_panderm":                             # PanDerm, the dermatology foundation model (ViT-L/16)
        m = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=1000, init_values=1e-5)
        missing, unexpected = m.load_state_dict(panderm_vision(), strict=False)
        assert not unexpected and set(missing) == head_keys(b), (missing, unexpected)
        return m
    if b in ("vits_dino", "vitb_dino"):                 # vitb_dino: second-DINO check only
        arch, ck = {"vits_dino": ("vit_small_patch16_224", "dino_deitsmall16_pretrain.pth"),
                    "vitb_dino": ("vit_base_patch16_224", "dino_vitbase16_pretrain.pth")}[b]
        m = timm.create_model(arch, pretrained=False)
        sd = torch.load(DATA / "dino" / ck, map_location="cpu")
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not unexpected and set(missing) == head_keys(b), (missing, unexpected)
        return m
    raise ValueError(b)


def hf_clip_vision(repo):
    """The vision tower of a Hugging Face CLIPModel checkpoint, as timm ViT keys (pre-norm CLIP, no head)."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    try:
        sd = load_file(hf_hub_download(repo, "model.safetensors"))
    except Exception:                                   # older repositories ship only the PyTorch pickle
        sd = torch.load(hf_hub_download(repo, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    p = "vision_model."
    out = {"cls_token": sd[p + "embeddings.class_embedding"].view(1, 1, -1),
           "pos_embed": sd[p + "embeddings.position_embedding.weight"].unsqueeze(0),
           "patch_embed.proj.weight": sd[p + "embeddings.patch_embedding.weight"],
           "norm_pre.weight": sd[p + "pre_layrnorm.weight"], "norm_pre.bias": sd[p + "pre_layrnorm.bias"],
           "norm.weight": sd[p + "post_layernorm.weight"], "norm.bias": sd[p + "post_layernorm.bias"]}
    n = 1 + max(int(k.split(".")[3]) for k in sd if k.startswith(p + "encoder.layers."))
    for i in range(n):
        q, b = f"{p}encoder.layers.{i}.", f"blocks.{i}."
        for w in ("weight", "bias"):
            out[f"{b}attn.qkv.{w}"] = torch.cat([sd[f"{q}self_attn.{x}_proj.{w}"] for x in "qkv"])
            for t, s in [("attn.proj", "self_attn.out_proj"), ("norm1", "layer_norm1"), ("norm2", "layer_norm2"),
                         ("mlp.fc1", "mlp.fc1"), ("mlp.fc2", "mlp.fc2")]:
                out[f"{b}{t}.{w}"] = sd[f"{q}{s}.{w}"]
    return out


def panderm_vision():
    """PanDerm's ViT-L/16 encoder (Yan et al. 2025; CC BY-NC-ND 4.0) as timm ViT keys. Its attention keeps q and v
    biases only (the k bias is zero), and its layer scales are gamma_1 / gamma_2."""
    sd = torch.load(DATA / "panderm" / "panderm_ll_data6_checkpoint-499.pth", map_location="cpu", weights_only=True)
    enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    out = {k.replace(".gamma_1", ".ls1.gamma").replace(".gamma_2", ".ls2.gamma"): v for k, v in enc.items()
           if not k.endswith(("attn.q_bias", "attn.v_bias"))}
    for k in [k for k in enc if k.endswith("attn.q_bias")]:
        q, v = enc[k], enc[k.replace("q_bias", "v_bias")]
        out[k.replace("q_bias", "qkv.bias")] = torch.cat([q, torch.zeros_like(q), v])
    return out


def norm_stats(b):
    return (CLIP_MEAN, CLIP_STD) if b in ("clip_b16", "vitl_clip", "vitl_monet") else (MEAN, STD)


def is_bn_stat(k):
    return k.startswith(("layer3.", "layer4.")) and k.endswith(("running_mean", "running_var"))


def is_ln_param(k):
    return k.startswith(tuple(f"blocks.{i}." for i in VIT_TRAINED)) and (".norm1." in k or ".norm2." in k)


def norm_param(b, k):
    return is_ln_param(k) if is_vit(b) else is_bn_stat(k)


def n_norm_params(b):
    return 4 * len(VIT_TRAINED) if is_vit(b) else 58


def stage_features(b, m, x):
    if not is_vit(b):
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        x = m.layer2(m.layer1(x)); s2 = x.mean((2, 3))
        x = m.layer3(x);           s3 = x.mean((2, 3))
        return {"s2": s2, "s3": s3, "s4": m.layer4(x).mean((2, 3))}
    if b == "vitl_dinov3_512":                          # timm's Eva: the rotary position embeddings go to every block
        x, rope = m._pos_embed(m.patch_embed(x))
        x, run = m.norm_pre(x), (lambda blk, x: blk(x, rope=rope))
    else:
        x, run = m.norm_pre(m.patch_drop(m._pos_embed(m.patch_embed(x)))), (lambda blk, x: blk(x))
    out = {}
    for i, blk in enumerate(m.blocks):
        x = run(blk, x)
        if f"b{i + 1}" in stages(b):
            out[f"b{i + 1}"] = x[:, m.num_prefix_tokens:].mean(1)
    out["cls"] = m.norm(x)[:, 0]
    return out


def family(cond):
    return cond.rsplit("_s", 1)[0]                  # "plant_shuf_s0" -> "plant_shuf"


def build(b, cond, ckpt_dir, sources):
    """The network for a condition. "imagenet" is the release itself and a run name loads ckpt_dir/<run>.pt. A
    norm-only control (ctrl(b) + a run of one of the sources) is the release with only that run's normalization
    parameters."""
    m = imagenet_model(b)
    if cond == "imagenet":
        return m
    c = ctrl(b)
    is_ctrl = cond.startswith(c) and family(cond)[len(c):] in sources
    run = cond[len(c):] if is_ctrl else cond
    sd = torch.load(ckpt_dir / f"{run}.pt", map_location="cpu")
    if is_ctrl:
        sd = {k: v for k, v in sd.items() if norm_param(b, k)}
        assert len(sd) == n_norm_params(b), (len(sd), n_norm_params(b))
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    if not is_ctrl:
        assert set(missing) == head_keys(b), missing
    return m
