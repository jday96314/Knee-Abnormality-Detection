#!/usr/bin/env python3
"""The public notebooks' two model families, ported.

Family A — a self-supervised ViT read one slot at a time, with the last few blocks
opened for training, and twelve learned queries attending over the six slot embeddings
of a study. This is the architecture behind the DINOv2 ensemble the community notebooks
submit, and behind the DINOv3 member added on top of it.

Family B — a frozen RadImageNet ResNet-50, one embedding per *slice* rather than per
slot, and a multi-head-attention query head over the resulting 24-token sequence. The
notebooks run this as an independent arm whose diversity is in the pretraining corpus
(radiology rather than natural images) rather than in the fold.

Both differ from architectures 1 and 2 in this repository in the same way: aggregation
happens over a handful of deliberately chosen images, not over every slice of every
series. Whether that choice is worth its cost is the question the sweep asks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

TIMM_NAMES = {
    "dinov2": "vit_small_patch14_dinov2.lvd142m",
    "dinov2b": "vit_base_patch14_dinov2.lvd142m",
    "dinov3": "vit_small_patch16_dinov3.lvd1689m",
}
POOL_PARTS = {"cls_mean": 2, "cls_mean_focal": 3}

# Which slots each finding's attention is tilted toward, as the notebooks fix it. Indices
# are into the six-slot recovered scheme. exp(0.55) gives a preferred slot about 1.73x
# the weight of an unpreferred one, which biases the softmax without excluding anything.
SLOT_PRIOR_TABLE = {
    "ACL": (0, 3, 5), "MCL": (1, 4),
    "Medial Meniscus": (0, 1, 3, 4), "Lateral Meniscus": (0, 1, 3, 4),
    "Medial OA": (1, 4, 5), "Lateral OA": (1, 4, 5),
    "PF OA": (0, 2, 5), "Effusion": (0, 2), "Synovitis": (0, 2),
    "Baker's": (0,), "Contusion": (0, 1, 2), "Fracture": (0, 1, 2, 4, 5),
}
SLOT_PRIOR_STRENGTH = 0.55

AUG_ROT_DEG = 8.0
AUG_SCALE = 0.08
AUG_SHIFT = 0.05
AUG_INTENSITY = 0.10


# --- augmentation ----------------------------------------------------------- #


def augment(imgs, generator=None):
    """Rigid jitter and an intensity scale, applied to a whole bag at once.

    Neither flip is available. A horizontal flip would reintroduce the nuisance axis the
    laterality normalisation exists to remove; a vertical flip is not a nuisance axis at
    all, since a knee is acquired in a canonical orientation and where a finding sits in
    the frame is information — a Baker's cyst is identified by lying in the popliteal
    fossa, not by its appearance alone.
    """
    lead = imgs.shape[:-3]
    x = imgs.reshape(-1, *imgs.shape[-3:]).float()
    n, dev = x.shape[0], x.device

    def rand(*shape):
        return torch.rand(*shape, device=dev, generator=generator)

    rot = (rand(n) - 0.5) * 2 * (AUG_ROT_DEG * np.pi / 180)
    # Zoom in only: `border` padding repeats the edge row outward, and the edge of this
    # crop is where the popliteal fossa sits, so zooming out would fabricate tissue in
    # exactly the place a Baker's cyst is looked for.
    sc = 1.0 + rand(n) * AUG_SCALE
    tx = (rand(n) - 0.5) * 2 * AUG_SHIFT
    ty = (rand(n) - 0.5) * 2 * AUG_SHIFT
    cos, sin = torch.cos(rot) / sc, torch.sin(rot) / sc
    theta = torch.zeros(n, 2, 3, device=dev, dtype=torch.float32)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

    scale = 1.0 + (rand(n, 1, 1, 1) - 0.5) * 2 * AUG_INTENSITY
    x = (x * scale).clamp(0, 255)
    return x.reshape(*lead, *x.shape[-3:]).to(imgs.dtype)


# --- family A: fine-tuned ViT + slot attention ------------------------------ #


class SlotHead(nn.Module):
    """Per-diagnosis attention over the slot embeddings of one study.

    Each finding is read on particular sequences — cruciates sagittally, collateral
    ligaments and the meniscal body coronally, patellar cartilage axially — so pooling
    the slots identically would dilute the one carrying the evidence with the rest.

    Deliberately this simple: with a study-level label there is no signal saying which
    part of a study matters, so attention parameters below the slot level would have
    nothing to learn from.
    """

    def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2, prior=False):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden
        self.prior = prior
        if prior:
            p_ = torch.zeros(n_out, n_slot)
            if n_slot == 6 and n_out == len(TARGETS):
                for t, slots in SLOT_PRIOR_TABLE.items():
                    p_[TARGETS.index(t), list(slots)] = SLOT_PRIOR_STRENGTH
            self.register_buffer("slot_prior", p_)

    def forward(self, x, mask):
        h = self.proj(x) + self.slot_emb
        att = torch.einsum("bsh,oh->bos", h, self.query) / self.hidden ** 0.5
        if self.prior:
            att = att + self.slot_prior.unsqueeze(0)
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))
        return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias


class MeanSlotHead(nn.Module):
    """The same head with the attention replaced by a masked mean over slots.

    The control for `SlotHead`: identical capacity everywhere except the one thing under
    test, so a difference between the two is the per-finding attention and not the
    projection, the slot embedding or the dropout.
    """

    def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2, prior=False):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)

    def forward(self, x, mask):
        h = self.proj(x) + self.slot_emb
        w = mask.unsqueeze(-1)
        ctx = (h * w).sum(1) / w.sum(1).clamp_min(1e-6)
        return self.out(self.drop(ctx))


class SlotModel(nn.Module):
    """Encoder plus head, trained end to end.

    A study arrives as a bag of slot images. The bag is flattened for the encoder and
    folded back before the head, so the encoder never sees the study structure and the
    head never sees pixels.
    """

    def __init__(self, backbone, dim, n_slot, pool="cls_mean", head="slot", prior=False,
                 n_prefix=1):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.n_prefix = n_prefix
        cls = SlotHead if head == "slot" else MeanSlotHead
        self.head = cls(dim * POOL_PARTS[pool], n_slot, len(TARGETS), prior=prior)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def encode(self, x):
        out = self.backbone.forward_features(x)
        patch = out[:, self.n_prefix:]
        parts = [out[:, 0], patch.mean(1)]
        if self.pool == "cls_mean_focal":
            # The upper tail of each channel over the patch grid, taken per channel
            # rather than by selecting whole patches: a finding occupies a small part of
            # the field, so a plain mean over hundreds of patches dilutes it.
            k = max(1, patch.shape[1] // 8)
            parts.append(patch.topk(k, dim=1).values.mean(1))
        return torch.cat(parts, dim=1)

    def forward(self, imgs, mask, img_size=None):
        B, S = imgs.shape[:2]
        x = imgs.reshape(B * S, *imgs.shape[2:]).float().div_(255.0)
        if img_size is not None and img_size != x.shape[-1]:
            # The cache is held at the highest resolution any configuration needs; the
            # rest downsample from it, so every configuration sees the same pixels
            # through a different sampling grid rather than a different crop.
            x = F.interpolate(x, size=(img_size, img_size), mode="bilinear",
                              align_corners=False)
        x = (x - self.mean) / self.std
        feat = self.encode(x).reshape(B, S, -1)
        return self.head(feat, mask)


def build_slot_model(arch="dinov2", img=224, n_slot=6, unfreeze_last=6, pool="cls_mean",
                     head="slot", prior=False):
    """Load the encoder and open the last `unfreeze_last` blocks for training.

    The early blocks of a self-supervised transformer are generic edge and texture
    filters; the late blocks carry semantics. Opening only the late ones is the cautious
    choice — there may not be enough supervision here to improve the early ones and
    there is certainly enough to damage them.
    """
    import timm

    bb = timm.create_model(TIMM_NAMES[arch], pretrained=True, num_classes=0, img_size=img)
    for prm in bb.parameters():
        prm.requires_grad = False
    for blk in bb.blocks[len(bb.blocks) - unfreeze_last:] if unfreeze_last else []:
        for prm in blk.parameters():
            prm.requires_grad = True
    for prm in bb.norm.parameters():
        prm.requires_grad = True
    return SlotModel(bb, bb.embed_dim, n_slot, pool=pool, head=head, prior=prior,
                     n_prefix=bb.num_prefix_tokens)


# --- family B: frozen RadImageNet + query head ------------------------------ #


RAD_CHECKPOINT_SHA256 = "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734"
RAD_TOKEN_DIM = 2048
RAD_HEAD_DIM = 512


def load_radimagenet(device="cuda"):
    """Strictly load the official RadImageNet ResNet-50 checkpoint.

    Checked by hash and by parameter count, as the notebook does: these weights are
    published in more than one namespace, and a partially-loaded ResNet produces
    features rather than an error.
    """
    import hashlib

    from torchvision.models import resnet50

    path = ASSETS / "ResNet50.pt"
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != RAD_CHECKPOINT_SHA256:
        raise RuntimeError(f"RadImageNet checkpoint drift: {digest.hexdigest()}")

    class RadImageNetEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(*list(resnet50(weights=None).children())[:-2])

        def forward(self, image):
            return self.backbone(image).mean(dim=(2, 3))

    model = RadImageNetEncoder()
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not state or not all(str(k).startswith("backbone.") for k in state):
        raise RuntimeError("unexpected RadImageNet state-dict namespace")
    model.load_state_dict(state, strict=True)
    n = sum(p.numel() for p in model.parameters())
    if n != 23_508_032:
        raise RuntimeError(f"unexpected RadImageNet parameter count {n}")
    return model.eval().to(device).requires_grad_(False)


class FoundationQueryHead(nn.Module):
    """Twelve queries cross-attending over one embedding per acquired slice.

    Unlike `SlotHead` this sees the slices individually — `n_slot * n_slice` tokens with
    a plane embedding and a within-stack position embedding added — so a finding can
    select a depth as well as a sequence.
    """

    def __init__(self, n_slot, n_slice, dim=RAD_TOKEN_DIM, hidden=RAD_HEAD_DIM,
                 n_out=len(TARGETS)):
        super().__init__()
        self.n_slot, self.n_slice, self.n_out = n_slot, n_slice, n_out
        self.project = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.plane = nn.Parameter(torch.randn(n_slot, hidden) * .01)
        self.position = nn.Parameter(torch.randn(n_slice, hidden) * .01)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * .02)
        self.attn = nn.MultiheadAttention(hidden, 8, dropout=.10, batch_first=True)
        self.fuse = nn.Sequential(
            nn.LayerNorm(hidden * 4), nn.Linear(hidden * 4, hidden),
            nn.GELU(), nn.Dropout(.15),
        )
        self.weight = nn.Parameter(torch.randn(n_out, hidden) * .02)
        self.bias = nn.Parameter(torch.zeros(n_out))

    def forward(self, feature, mask):
        token = self.project(feature.float())
        token = token.view(len(token), self.n_slot, self.n_slice, -1)
        token = token + self.plane[None, :, None] + self.position[None, None]
        token = token.flatten(1, 2)
        key_padding = mask <= 0
        # No study should be empty, but keep the attention numerically defined if one is.
        all_empty = key_padding.all(1)
        if all_empty.any():
            key_padding = key_padding.clone()
            key_padding[all_empty, 0] = False
        query = self.query.unsqueeze(0).expand(len(token), -1, -1)
        attended = query + self.attn(query, token, token,
                                     key_padding_mask=key_padding, need_weights=False)[0]
        denom = mask.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)
        mean = (token * mask.unsqueeze(-1)).sum(1, keepdims=True) / denom
        mean = mean.expand(-1, self.n_out, -1)
        fused = self.fuse(torch.cat(
            [attended, mean, torch.abs(attended - mean), attended * mean], -1))
        return (fused * self.weight.unsqueeze(0)).sum(-1) + self.bias
