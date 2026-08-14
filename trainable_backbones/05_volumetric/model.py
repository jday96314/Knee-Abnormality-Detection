#!/usr/bin/env python3
"""Architectures 5 and 6: pretrained 3D encoders applied per series.

The plan lists these separately — #5 "pretrained 3D MRI encoder", #6 "3D/video/CoPAS-style
diversity branch" — but structurally they are one family: a volumetric encoder turns a whole
series into one vector, and the study head is unchanged. The only difference is which
pretraining the encoder carries, so they live in one directory and differ by `--encoder`:

  * `medicalnet`  MONAI's 3D ResNet-18 with MedicalNet weights — 3D medical volumes,
                  including MRI. This is the plan's wildcard: the one encoder whose
                  pretraining actually matches the input.
  * `r3d18`       torchvision's ResNet-3D-18 pretrained on Kinetics-400 video. The
                  diversity branch. Kinetics has nothing to do with knees, but its
                  inductive bias — motion/continuity along the third axis — is a genuinely
                  different prior from a per-slice 2D encoder.

Both are frozen first, per the plan's instruction not to train a large 3D network on 4,000
noisy labels, with partial unfreezing available as an option.
"""

from __future__ import annotations

import torch
import torch.nn as nn

PLANES = ["Sagittal", "Coronal", "Axial"]
PLANE_INDEX = {p: i for i, p in enumerate(PLANES)}


class VolumeEncoder(nn.Module):
    """[N, 1, D, H, W] volume -> [N, dim].

    Grayscale is replicated to three channels for the video model, which is the same
    compromise the 2D encoders make and is unavoidable for weights trained on RGB.
    """

    def __init__(self, kind: str = "medicalnet", unfreeze: int = 0):
        super().__init__()
        if kind == "medicalnet":
            from monai.networks.nets import resnet18
            self.net = resnet18(spatial_dims=3, n_input_channels=1, feed_forward=False,
                                shortcut_type="A", bias_downsample=True, pretrained=True)
            self.dim, self.in_channels = 512, 1
            stages = [self.net.layer1, self.net.layer2, self.net.layer3, self.net.layer4]
        elif kind == "r3d18":
            from torchvision.models.video import R3D_18_Weights, r3d_18
            self.net = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
            self.net.fc = nn.Identity()
            self.dim, self.in_channels = 512, 3
            stages = [self.net.layer1, self.net.layer2, self.net.layer3, self.net.layer4]
        else:
            raise ValueError(f"unknown encoder {kind}")

        for p in self.net.parameters():
            p.requires_grad_(False)
        for stage in stages[len(stages) - unfreeze:] if unfreeze else []:
            for p in stage.parameters():
                p.requires_grad_(True)
        self.kind, self.unfrozen = kind, unfreeze

    def forward(self, x):
        if self.in_channels == 3:
            x = x.repeat(1, 3, 1, 1, 1)
        out = self.net(x)
        return out.flatten(1)


class VolumetricModel(nn.Module):
    """Per-series volumes -> 12 logits.

    The study head is architecture 1's winner (fixed per-plane mean/max) rather than the
    plan's 12-query attention, because query attention lost on frozen features and lost
    again in architecture 3. Reusing the settled head keeps the encoder as the only thing
    being tested here.
    """

    def __init__(self, kind: str = "medicalnet", unfreeze: int = 0, d_model: int = 256,
                 dropout: float = 0.1, n_labels: int = 12):
        super().__init__()
        self.encoder = VolumeEncoder(kind, unfreeze)
        self.project = nn.Sequential(nn.LayerNorm(self.encoder.dim),
                                     nn.Linear(self.encoder.dim, d_model), nn.GELU())
        self.plane_emb = nn.Embedding(len(PLANES) + 1, d_model)
        self.fluid_emb = nn.Embedding(2, d_model)
        self.fatsat_emb = nn.Embedding(2, d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * len(PLANES) * d_model), nn.Dropout(dropout),
            nn.Linear(2 * len(PLANES) * d_model, n_labels))

    def forward(self, volumes, plane, fluid, fatsat, mask):
        """volumes [B, S, D, H, W]; mask [B, S]."""
        B, S = volumes.shape[:2]
        flat = volumes.reshape(B * S, 1, *volumes.shape[2:])
        tokens = self.project(self.encoder(flat)).reshape(B, S, -1)
        tokens = tokens + self.plane_emb(plane) + self.fluid_emb(fluid) + self.fatsat_emb(fatsat)
        chunks = []
        for p in range(len(PLANES)):
            sel = mask & (plane == p)
            m = sel.unsqueeze(-1).float()
            chunks.append((tokens * m).sum(1) / m.sum(1).clamp(min=1))
            chunks.append(torch.nan_to_num(
                tokens.masked_fill(~sel.unsqueeze(-1), float("-inf")).max(1).values, neginf=0.0))
        return self.head(torch.cat(chunks, dim=-1))
