#!/usr/bin/env python3
"""Frozen feature extractors: MRI-CORE (ViT-B/16) and OrthoFoundation (DINOv3 ViT-L/16).

Each backbone exposes the same contract: take a batch of ImageNet-normalized
[B, 3, H, W] float tensors and return a dict of named [B, D] descriptors. Both
models are wrapped rather than subclassed because their repos disagree about
everything except the normalization constants.

Both run in bfloat16, not float16. DINOv3 overflows in fp16 -- every feature
comes back NaN -- which is unsurprising given it is trained with bf16 layer
norms; bf16 and fp32 read-outs agree to ~1e-2.

Several descriptors are returned per slice rather than one. Which read-out of a
frozen encoder linear-probes best is not knowable in advance -- a CLS token and
a patch mean answer to different training signals -- so the choice is deferred
to the probe rather than fixed here.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent

# Both repos were trained on ImageNet-normalized inputs: MRI-CORE via
# utils/dataset.py (min-max to [0,1] then Normalize), DINOv3 by convention.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _spatial_stats(feature_map: torch.Tensor) -> dict[str, torch.Tensor]:
    """Mean, max and std over the spatial axes of a [B, C, H, W] map.

    Kept separate rather than concatenated so the probe can weigh them
    independently: the mean describes the slice, the max reports the single most
    responsive location, and the std says how uneven the slice is. A focal
    finding on an otherwise normal knee moves the last two and barely touches
    the first.
    """
    flat = feature_map.flatten(2)
    return {
        "mean": flat.mean(dim=2),
        "max": flat.amax(dim=2),
        "std": flat.std(dim=2),
    }


class MRICore(nn.Module):
    """MRI-CORE: a DINO-pretrained plain ViT-B/16, loaded outside its own repo.

    The released checkpoint is the DINO *teacher* -- a plain ViT-B/16 at 224px
    (`pos_embed` is [1, 197, 768], there is no layerscale, and the blocks sit in
    DINOv2 `BlockChunk`s). It is not a SAM model, despite the README routing
    feature extraction through `sam_model_registry`. That path is avoided here
    for two reasons:

      1. `_build_sam` writes the interpolated position embedding back under its
         source key (`backbone.pos_embed`) instead of the remapped one
         (`image_encoder.pos_embed`), so `pos_embed` silently stays at random
         init. Confirmed: loading that way reports `image_encoder.pos_embed` as
         missing and `backbone.pos_embed` as unexpected.
      2. SAM's `ImageEncoderViT` runs windowed attention on all but four blocks
         and appends a randomly-initialized neck -- the checkpoint contains no
         neck weights at all. Neither matches how these weights were trained.

    Loading the same tensors into a plain ViT reproduces the encoder that DINO
    actually trained. `dynamic_img_size` lets the 224px position embedding be
    interpolated to a larger grid, which is a resolution knob the probe can test
    rather than a correction.
    """

    name = "mri_core"

    def __init__(self, checkpoint: Path, device: str = "cuda", dtype=torch.bfloat16,
                 input_size: int = 224):
        super().__init__()
        import timm

        self.input_size = input_size
        # Built at the checkpoint's native 224 so pos_embed loads at its trained
        # shape; dynamic_img_size then resamples it per-forward, so a larger
        # input_size costs nothing at load time.
        model = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=0,
            img_size=224, dynamic_img_size=True,
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["teacher"]
        remapped = {}
        for key, value in state.items():
            if not key.startswith("backbone.") or "mask_token" in key:
                continue  # mask_token and the dino_head are pretraining-only
            key = key.removeprefix("backbone.")
            # BlockChunk pads each chunk with Identity so the within-chunk index
            # is already the global block index; only the chunk id is dropped.
            key = re.sub(r"^blocks\.\d+\.(\d+)\.", r"blocks.\1.", key)
            remapped[key] = value
        # strict=True: 0 missing / 0 unexpected is the evidence the arch matches.
        model.load_state_dict(remapped, strict=True)

        self.model = model.to(device=device, dtype=dtype).eval()
        self.device, self.dtype = device, dtype
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        images = images.to(self.device, self.dtype)
        tokens = self.model.forward_features(images).float()  # [B, 1+N, 768]
        cls, patches = tokens[:, 0], tokens[:, 1:]

        side = int(patches.shape[1] ** 0.5)
        grid = patches.transpose(1, 2).reshape(patches.shape[0], -1, side, side)
        stats = _spatial_stats(grid)
        return {"cls": cls, **{f"patch_{k}": v for k, v in stats.items()}}


class OrthoFoundation(nn.Module):
    """OrthoFoundation-L: DINOv3 ViT-L/16 continued-pretrained on knee imaging.

    The released checkpoint stores only the student backbone under a `backbone.`
    prefix, so it is loaded into the reference DINOv3 architecture rather than
    through the repo's torch.hub path, which expects DINOv3's gated weights to
    be present as a starting point.
    """

    name = "orthofoundation"
    input_size = 448  # 28x28 patches; DINOv3 handles arbitrary multiples of 16

    def __init__(self, checkpoint: Path, device: str = "cuda", dtype=torch.bfloat16,
                 input_size: int | None = None):
        super().__init__()
        if input_size is not None:
            self.input_size = input_size
        sys.path.insert(0, str(HERE / "dinov3"))
        from dinov3.models.vision_transformer import vit_large

        # These are the DINOv3 ViT-L/16 defaults; the checkpoint loads with zero
        # missing and zero unexpected keys, which is the check that they match.
        model = vit_large(
            patch_size=16,
            n_storage_tokens=4,
            qkv_bias=True,
            mask_k_bias=True,
            layerscale_init=1e-5,
            norm_layer="layernormbf16",
            ffn_layer="mlp",
            pos_embed_rope_base=100.0,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2.0,
            pos_embed_rope_dtype="fp32",
            untie_cls_and_patch_norms=False,
            untie_global_and_local_cls_norm=False,
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = {k.removeprefix("backbone."): v for k, v in state.items()
                 if k.startswith("backbone.")}
        model.load_state_dict(state, strict=True)
        self.model = model.to(device=device, dtype=dtype).eval()
        self.device, self.dtype = device, dtype
        self.embed_dim = model.embed_dim
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        images = images.to(self.device, self.dtype)
        out = self.model.forward_features(images)
        cls = out["x_norm_clstoken"].float()  # [B, 1024]
        patches = out["x_norm_patchtokens"].float()  # [B, N, 1024]

        side = int(patches.shape[1] ** 0.5)
        grid = patches.transpose(1, 2).reshape(patches.shape[0], -1, side, side)
        stats = _spatial_stats(grid)
        return {"cls": cls, **{f"patch_{k}": v for k, v in stats.items()}}


def load_backbone(name: str, device: str = "cuda", **kwargs) -> nn.Module:
    if name == "mri_core":
        return MRICore(HERE / "mri_foundation/pretrained_weights/MRI_CORE_vitb.pth",
                       device=device, **kwargs)
    if name == "orthofoundation":
        return OrthoFoundation(HERE / "OrthoFoundation/OrthoFoudation-L.pth",
                               device=device, **kwargs)
    raise ValueError(f"unknown backbone: {name}")
