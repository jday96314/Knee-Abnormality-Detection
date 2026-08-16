#!/usr/bin/env python3
"""Trainable pretrained encoders for architecture 3.

One wrapper over four families with incompatible interfaces, so `train.py` only ever sees
`[N, 1, H, W] -> [N, dim]` and an integer "how many top units to unfreeze".

  mri_core      MRI-CORE ViT-B/16      12 blocks   768  MRI, self-supervised (DINO)
  ortho         OrthoFoundation ViT-L  24 blocks  1024  musculoskeletal, DINOv3
  radimagenet   RadImageNet ResNet-50   4 stages  2048  radiology, supervised
  dinov2s       DINOv2 ViT-S/14        12 blocks   384  natural images, self-supervised
  dinov3s       DINOv3 ViT-S/16        12 blocks   384  natural images, self-supervised

`unfreeze` counts *units* from the top — blocks for the transformers, residual stages for
the ResNet — so it is comparable in intent across families but not in parameter count. All
are left in float32; bfloat16 master weights and Adam do not mix well at this scale, and
autocast handles the arithmetic.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "frozen_backbones"
RAD_WEIGHTS = ROOT / "trainable_backbones" / "07_public" / "assets" / "ResNet50.pt"
RAD_SHA256 = None  # set from the sibling module's constant at load time, if available

NATIVE_SIZE = {"mri_core": 224, "ortho": 224, "radimagenet": 224,
               "dinov2s": 224, "dinov3s": 224}
N_UNITS = {"mri_core": 12, "ortho": 24, "radimagenet": 4, "dinov2s": 12, "dinov3s": 12}


class TrainableEncoder(nn.Module):
    """[N, 1, H, W] grayscale -> [N, dim] pooled embedding, top `unfreeze` units trainable."""

    def __init__(self, name: str = "mri_core", unfreeze: int = 4, device: str = "cuda"):
        super().__init__()
        self.name, self.unfrozen = name, unfreeze
        build = {
            "mri_core": self._mri_core, "ortho": self._ortho,
            "radimagenet": self._radimagenet,
            "dinov2s": lambda: self._timm_vit("vit_small_patch14_dinov2.lvd142m"),
            "dinov3s": lambda: self._timm_vit("vit_small_patch16_dinov3.lvd1689m"),
        }
        if name not in build:
            raise ValueError(f"unknown backbone {name}; have {sorted(build)}")
        units = build[name]()                                  # list of top-level units
        for p in self.model.parameters():
            p.requires_grad_(False)
        for unit in units[len(units) - unfreeze:] if unfreeze else []:
            for p in unit.parameters():
                p.requires_grad_(True)
        for tail in self._tail():
            for p in tail.parameters():
                p.requires_grad_(True)
        self.model = self.model.to(device)

    # --- families ---------------------------------------------------------- #

    def _mri_core(self):
        import timm
        model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0,
                                  img_size=224, dynamic_img_size=True)
        state = torch.load(FROZEN / "mri_foundation/pretrained_weights/MRI_CORE_vitb.pth",
                           map_location="cpu", weights_only=False)["teacher"]
        remapped = {}
        for key, value in state.items():
            if not key.startswith("backbone.") or "mask_token" in key:
                continue
            key = re.sub(r"^blocks\.\d+\.(\d+)\.", r"blocks.\1.", key.removeprefix("backbone."))
            remapped[key] = value
        model.load_state_dict(remapped, strict=True)
        self.model, self.dim, self.kind = model, model.embed_dim, "timm_vit"
        return list(model.blocks)

    def _timm_vit(self, tag: str):
        import timm
        model = timm.create_model(tag, pretrained=True, num_classes=0,
                                  img_size=224, dynamic_img_size=True)
        self.model, self.dim, self.kind = model, model.embed_dim, "timm_vit"
        return list(model.blocks)

    def _ortho(self):
        """DINOv3 ViT-L/16 from the OrthoFoundation checkpoint.

        Loaded directly rather than through `frozen_backbones.backbones`, whose wrapper
        pins bfloat16, calls `.eval()` and wraps `forward` in `torch.no_grad` — all three
        are correct for a frozen probe and fatal for fine-tuning.
        """
        sys.path.insert(0, str(FROZEN / "dinov3"))
        from dinov3.models.vision_transformer import vit_large
        model = vit_large(
            patch_size=16, n_storage_tokens=4, qkv_bias=True, mask_k_bias=True,
            layerscale_init=1e-5, norm_layer="layernormbf16", ffn_layer="mlp",
            pos_embed_rope_base=100.0, pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2.0, pos_embed_rope_dtype="fp32",
            untie_cls_and_patch_norms=False, untie_global_and_local_cls_norm=False,
        )
        state = torch.load(FROZEN / "OrthoFoundation/OrthoFoudation-L.pth",
                           map_location="cpu", weights_only=True)
        model.load_state_dict({k.removeprefix("backbone."): v for k, v in state.items()
                               if k.startswith("backbone.")}, strict=True)
        self.model, self.dim, self.kind = model.float(), model.embed_dim, "dinov3"
        return list(model.blocks)

    def _radimagenet(self):
        """RadImageNet ResNet-50, verified by hash and parameter count.

        These weights are published in more than one namespace and a partially-loaded
        ResNet yields plausible features rather than an error, so both checks are load
        errors rather than warnings.
        """
        from torchvision.models import resnet50
        state = torch.load(RAD_WEIGHTS, map_location="cpu", weights_only=True)
        if not state or not all(str(k).startswith("backbone.") for k in state):
            raise RuntimeError("unexpected RadImageNet state-dict namespace")
        model = nn.Sequential(*list(resnet50(weights=None).children())[:-2])
        model.load_state_dict({k.removeprefix("backbone."): v for k, v in state.items()},
                              strict=True)
        n = sum(p.numel() for p in model.parameters())
        if n != 23_508_032:
            raise RuntimeError(f"unexpected RadImageNet parameter count {n}")
        digest = hashlib.sha256()
        with open(RAD_WEIGHTS, "rb") as fh:
            for chunk in iter(lambda: fh.read(8 << 20), b""):
                digest.update(chunk)
        self.sha256 = digest.hexdigest()
        self.model, self.dim, self.kind = model, 2048, "resnet"
        return [model[4], model[5], model[6], model[7]]        # layer1..layer4

    def _tail(self):
        """Normalisation layers that should always train, being cheap and input-dependent."""
        if self.kind in ("timm_vit", "dinov3") and hasattr(self.model, "norm"):
            return [self.model.norm]
        return []

    # --- forward ----------------------------------------------------------- #

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images.repeat(1, 3, 1, 1) if images.shape[1] == 1 else images
        if self.kind == "timm_vit":
            return self.model.forward_features(x)[:, 0]
        if self.kind == "dinov3":
            return self.model.forward_features(x)["x_norm_clstoken"]
        return self.model(x).mean(dim=(2, 3))                  # resnet: global average pool


def trainable_report(encoder: TrainableEncoder) -> str:
    total = sum(p.numel() for p in encoder.parameters())
    train = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    return (f"{encoder.name}: dim {encoder.dim}, {train/1e6:.1f}M trainable "
            f"of {total/1e6:.1f}M ({encoder.unfrozen}/{N_UNITS[encoder.name]} units)")
