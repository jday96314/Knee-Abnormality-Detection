#!/usr/bin/env python3
"""DICOM loading and image preparation for the image-only MedGemma experiments.

Nothing here reads labels or report text. The module turns a study folder into an
ordered set of PNG data URIs plus the series metadata needed to caption them.

The archive is unpacked incrementally, so `complete_labeled_studies` deliberately
admits only studies whose on-disk series match `train_series.csv` exactly. Results
computed on a partial cohort are not comparable to a later, larger cohort, so the
cohort is hashed and recorded by the caller.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pydicom
from PIL import Image


LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


@dataclass(frozen=True)
class SeriesRef:
    """One selected acquisition, with the descriptors the prompt is allowed to see."""

    uid: str
    series_uid: str
    plane: str
    fluid_sensitive: int
    fat_suppression: int
    role: str

    def caption(self) -> str:
        weighting = (
            "fluid-sensitive (T2/PD/STIR-like)"
            if self.fluid_sensitive
            else "non-fluid-sensitive (T1-like)"
        )
        fat = "fat-suppressed" if self.fat_suppression else "no fat suppression"
        return f"{self.plane} plane, {weighting}, {fat}"


# ---------------------------------------------------------------------------
# Cohort discovery
# ---------------------------------------------------------------------------


def complete_labeled_studies(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (labeled studies fully present on disk, their series metadata).

    A study counts as complete only when every series listed in `train_series.csv`
    exists on disk and is non-empty. Partially extracted studies are dropped rather
    than silently scored on fewer images than intended.
    """
    train = pd.read_csv(data_dir / "train.csv")
    series = pd.read_csv(data_dir / "train_series.csv")
    root = data_dir / "train_series"

    labeled = train.dropna(subset=LABELS).copy()
    expected = series.groupby("StudyInstanceUID")["SeriesInstanceUID"].apply(set)

    keep: list[str] = []
    for uid in labeled["StudyInstanceUID"]:
        folder = root / uid
        if not folder.is_dir():
            continue
        on_disk = {p.name for p in folder.iterdir() if p.is_dir()}
        if on_disk != expected.get(uid, set()):
            continue
        if any(not any((folder / s).iterdir()) for s in on_disk):
            continue
        keep.append(uid)

    labeled = labeled[labeled["StudyInstanceUID"].isin(keep)].reset_index(drop=True)
    for label in LABELS:
        labeled[label] = labeled[label].astype(int)
    series = series[series["StudyInstanceUID"].isin(keep)].reset_index(drop=True)
    return labeled, series


def cohort_id(uids: list[str]) -> str:
    """Stable short hash of the scored cohort, so runs on different cohorts never merge."""
    digest = hashlib.sha256("|".join(sorted(uids)).encode()).hexdigest()
    return f"n{len(uids)}_{digest[:10]}"


# ---------------------------------------------------------------------------
# Series selection
# ---------------------------------------------------------------------------

# Ordered preferences. Each entry is (role, plane, fluid_sensitive or None).
# Fluid-sensitive sagittal/coronal/axial carry most of the pathology signal for
# effusion, contusion, meniscus and cartilage; one T1-like series adds anatomy.
SERIES_PLANS: dict[str, list[tuple[str, str, int | None]]] = {
    "sag_fluid": [("sagittal_fluid", "Sagittal", 1)],
    "tri_fluid": [
        ("sagittal_fluid", "Sagittal", 1),
        ("coronal_fluid", "Coronal", 1),
        ("axial_fluid", "Axial", 1),
    ],
    "tri_fluid_plus_t1": [
        ("sagittal_fluid", "Sagittal", 1),
        ("coronal_fluid", "Coronal", 1),
        ("axial_fluid", "Axial", 1),
        ("sagittal_t1", "Sagittal", 0),
    ],
    "all_planes_any": [
        ("sagittal_any", "Sagittal", None),
        ("coronal_any", "Coronal", None),
        ("axial_any", "Axial", None),
    ],
    "cor_fluid": [("coronal_fluid", "Coronal", 1)],
    "ax_fluid": [("axial_fluid", "Axial", 1)],
    "sag_cor_fluid": [
        ("sagittal_fluid", "Sagittal", 1),
        ("coronal_fluid", "Coronal", 1),
    ],
    "sag_ax_fluid": [
        ("sagittal_fluid", "Sagittal", 1),
        ("axial_fluid", "Axial", 1),
    ],
}


# Which plane actually answers each question. A cruciate ligament is a sagittal
# problem, a collateral ligament a coronal one, a Baker's cyst an axial one; showing
# all four series for every finding spends tokens on views that cannot resolve it.
FINDING_PLAN = {
    "ACL": "sag_fluid",
    "MCL": "cor_fluid",
    "Medial Meniscus": "sag_cor_fluid",
    "Lateral Meniscus": "sag_cor_fluid",
    "Medial OA": "cor_fluid",
    "Lateral OA": "cor_fluid",
    "PF OA": "sag_ax_fluid",
    "Effusion": "sag_ax_fluid",
    "Synovitis": "sag_ax_fluid",
    "Baker's": "ax_fluid",
    "Contusion": "sag_cor_fluid",
    "Fracture": "sag_cor_fluid",
}


def _all_series(rows: pd.DataFrame, uid: str) -> list[SeriesRef]:
    """Every series in the study, ordered by plane then UID.

    Studies carry a median of 5 series and up to 14, so this is the regime where a
    study can outgrow the context window; the role-based plans never reach it.
    """
    order = {"Sagittal": 0, "Coronal": 1, "Axial": 2}
    ordered = rows.assign(_plane=rows["Anatomical_Plane"].map(order).fillna(9)).sort_values(
        ["_plane", "SeriesInstanceUID"]
    )
    return [
        SeriesRef(
            uid=uid,
            series_uid=row["SeriesInstanceUID"],
            plane=row["Anatomical_Plane"],
            fluid_sensitive=int(row["Fluid_Sensitive"]),
            fat_suppression=int(row["Fat_Suppression"]),
            role=f"{str(row['Anatomical_Plane']).lower()}_all",
        )
        for _, row in ordered.iterrows()
    ]


def select_series(series: pd.DataFrame, uid: str, plan: str) -> list[SeriesRef]:
    """Pick one series per requested role, preferring fat-suppressed fluid-sensitive scans."""
    rows = series[series["StudyInstanceUID"] == uid]
    if plan == "all_series":
        return _all_series(rows, uid)

    chosen: list[SeriesRef] = []
    used: set[str] = set()

    for role, plane, fluid in SERIES_PLANS[plan]:
        candidates = rows[rows["Anatomical_Plane"] == plane]
        if fluid is not None:
            exact = candidates[candidates["Fluid_Sensitive"] == fluid]
            candidates = exact if len(exact) else candidates
        candidates = candidates[~candidates["SeriesInstanceUID"].isin(used)]
        if not len(candidates):
            continue
        # Deterministic tie-break: fat suppression first, then UID order.
        candidates = candidates.sort_values(
            ["Fat_Suppression", "SeriesInstanceUID"], ascending=[False, True]
        )
        row = candidates.iloc[0]
        used.add(row["SeriesInstanceUID"])
        chosen.append(
            SeriesRef(
                uid=uid,
                series_uid=row["SeriesInstanceUID"],
                plane=row["Anatomical_Plane"],
                fluid_sensitive=int(row["Fluid_Sensitive"]),
                fat_suppression=int(row["Fat_Suppression"]),
                role=role,
            )
        )
    return chosen


# ---------------------------------------------------------------------------
# Slice ordering and rendering
# ---------------------------------------------------------------------------


ORDER_CACHE_DIR: Path | None = None


def set_order_cache(path: Path | None) -> None:
    """Where to persist slice orderings between processes and runs."""
    global ORDER_CACHE_DIR
    ORDER_CACHE_DIR = path
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=4096)
def ordered_slice_paths(series_dir: str) -> tuple[str, ...]:
    """Slice file paths in through-plane order.

    InstanceNumber is used when it is present and unique; otherwise slices are
    ordered by their position projected onto the slice normal.

    Establishing the order means reading a header from every slice in the series —
    on the order of a hundred reads per study, which on a slow disk costs more than
    decoding the handful of slices actually wanted. The result depends only on the
    files, so it is memoised to disk as well as in process memory.
    """
    if ORDER_CACHE_DIR is not None:
        digest = hashlib.sha256(series_dir.encode()).hexdigest()[:20]
        cache_file = ORDER_CACHE_DIR / f"{digest}.json"
        if cache_file.exists():
            try:
                return tuple(json.loads(cache_file.read_text()))
            except (json.JSONDecodeError, OSError):
                pass  # a corrupt or half-written entry is simply recomputed
        order = _compute_slice_order(series_dir)
        tmp = cache_file.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(order))
        os.replace(tmp, cache_file)
        return order
    return _compute_slice_order(series_dir)


def _compute_slice_order(series_dir: str) -> tuple[str, ...]:
    paths = [str(p) for p in sorted(Path(series_dir).iterdir()) if p.suffix == ".dcm"]
    keys: list[tuple[float, str]] = []
    numbers: list[float] = []
    positions: list[float] = []

    for path in paths:
        header = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        numbers.append(float(getattr(header, "InstanceNumber", "nan") or "nan"))
        orientation = getattr(header, "ImageOrientationPatient", None)
        position = getattr(header, "ImagePositionPatient", None)
        if orientation is not None and position is not None:
            row_vec = np.array(orientation[:3], dtype=float)
            col_vec = np.array(orientation[3:], dtype=float)
            normal = np.cross(row_vec, col_vec)
            positions.append(float(np.dot(np.array(position, dtype=float), normal)))
        else:
            positions.append(float("nan"))

    use_numbers = not any(np.isnan(numbers)) and len(set(numbers)) == len(numbers)
    order = numbers if use_numbers else positions
    if any(np.isnan(order)):
        return tuple(paths)
    keys = sorted(zip(order, paths))
    return tuple(path for _, path in keys)


def sample_indices(
    n_slices: int, n_take: int, center_fraction: float = 0.7, phase: float = 0.0
) -> list[int]:
    """Evenly spaced indices from the central `center_fraction` of the stack.

    The first and last slices of a knee series are usually off-joint, so the
    default trims them before sampling.

    `phase` shifts the sampling grid by that fraction of one inter-sample step. It is
    the basis of slice-selection test-time augmentation: the same series viewed
    through a different set of slices, which for a sparse subsample is a genuinely
    different look at the joint rather than a re-roll of the same one.
    """
    if n_take >= n_slices:
        return list(range(n_slices))
    span = max(1.0, n_slices * center_fraction)
    start = (n_slices - span) / 2.0
    return [
        int(round(start + span * min(i + 0.5 + phase, n_take - 0.01) / n_take))
        for i in range(n_take)
    ]


def slab_bounds(n_slices: int, n_slabs: int, center_fraction: float) -> list[tuple[int, int]]:
    """Contiguous [start, stop) slabs partitioning the central region of the stack."""
    span = max(1.0, n_slices * center_fraction)
    start = (n_slices - span) / 2.0
    edges = [int(round(start + span * i / n_slabs)) for i in range(n_slabs + 1)]
    bounds = []
    for lower, upper in zip(edges, edges[1:]):
        lower = max(0, min(lower, n_slices - 1))
        upper = max(lower + 1, min(upper, n_slices))
        bounds.append((lower, upper))
    return bounds


def planned_take(n_slices: int, spec: "ImageSpec") -> int:
    """How many images one series contributes, before any study-level budget cap.

    `fixed` ignores series length entirely, so a 320-slice series and a 20-slice one
    are represented by the same number of pictures — a 16x difference in through-plane
    coverage. `proportional` instead keeps the sampling density constant by taking
    roughly every `stride`-th slice, which is the comparison this axis exists to make.
    """
    if spec.sampling == "proportional":
        usable = max(1.0, n_slices * spec.center_fraction)
        return int(min(spec.slices_per_series, max(2, round(usable / spec.stride))))
    return spec.slices_per_series


def _window(array: np.ndarray, header: Any, mode: str) -> np.ndarray:
    array = array.astype(np.float32)
    slope = float(getattr(header, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(header, "RescaleIntercept", 0.0) or 0.0)
    array = array * slope + intercept

    center = getattr(header, "WindowCenter", None)
    width = getattr(header, "WindowWidth", None)
    if mode == "dicom" and center is not None and width is not None:
        center = float(center[0] if hasattr(center, "__iter__") else center)
        width = max(float(width[0] if hasattr(width, "__iter__") else width), 1.0)
        low, high = center - width / 2.0, center + width / 2.0
    else:
        low, high = np.percentile(array, [1.0, 99.0])

    array = np.clip((array - low) / max(high - low, 1e-6), 0.0, 1.0)
    if str(getattr(header, "PhotometricInterpretation", "MONOCHROME2")) == "MONOCHROME1":
        array = 1.0 - array
    return array


def render_slice(path: str, size: int, window: str, flip: bool = False) -> Image.Image:
    """Decode one slice to a square 8-bit grayscale image, preserving aspect ratio."""
    dataset = pydicom.dcmread(path, force=True)
    pixels = dataset.pixel_array
    if pixels.ndim == 3:  # rare multi-frame or colour slice
        pixels = pixels[pixels.shape[0] // 2] if pixels.shape[-1] > 4 else pixels[..., 0]
    array = _window(pixels, dataset, window)
    image = Image.fromarray((array * 255.0).astype(np.uint8), mode="L")

    # Letterbox rather than stretch: knee series are often non-square and
    # distorting them changes apparent cartilage and meniscus geometry.
    scale = size / max(image.size)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.BILINEAR,
    )
    canvas = Image.new("L", (size, size), color=0)
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas.transpose(Image.FLIP_LEFT_RIGHT) if flip else canvas


def reduce_slab(paths: list[str], size: int, window: str, mode: str, flip: bool = False) -> Image.Image:
    """Collapse a contiguous slab of slices into one image.

    `slab_mip` takes the per-pixel maximum, the standard way to keep a small bright
    finding — effusion, a cyst, marrow oedema on a fluid-sensitive sequence — that a
    sparse subsample would step straight over. `slab_mean` averages instead, which
    suppresses noise but dilutes exactly those focal findings.
    """
    stack = np.stack(
        [np.asarray(render_slice(path, size, window, flip), dtype=np.float32) for path in paths]
    )
    collapsed = stack.max(axis=0) if mode == "slab_mip" else stack.mean(axis=0)
    return Image.fromarray(collapsed.astype(np.uint8), mode="L")


def montage(images: list[Image.Image], size: int) -> Image.Image:
    """Tile slices into one square image, in reading order."""
    columns = int(np.ceil(np.sqrt(len(images))))
    rows = int(np.ceil(len(images) / columns))
    cell = size // max(columns, rows)
    canvas = Image.new("L", (cell * columns, cell * rows), color=0)
    for index, image in enumerate(images):
        canvas.paste(image.resize((cell, cell), Image.BILINEAR),
                     ((index % columns) * cell, (index // columns) * cell))
    return canvas


def png_data_uri(make_image: Callable[[], Image.Image], cache_path: Path | None = None) -> str:
    """Encode an image as a data URI, rendering it only if it is not already cached.

    The image is passed as a thunk rather than a value so that a cache hit skips the
    DICOM decode as well as the PNG encode. Decoding is by far the expensive half —
    several seconds per study — so evaluating it eagerly made a warm cache almost
    worthless.
    """
    payload = b""
    if cache_path is not None and cache_path.exists():
        payload = cache_path.read_bytes()
        # A truncated entry is worse than a missing one: the server rejects it with an
        # opaque "cannot identify image file" and the cell fails for every finding that
        # uses that study. Cheap to detect here from the PNG magic and IEND trailer.
        if not (payload.startswith(b"\x89PNG\r\n\x1a\n") and payload.endswith(b"IEND\xaeB`\x82")):
            payload = b""

    if not payload:
        buffer = io.BytesIO()
        make_image().save(buffer, format="PNG", optimize=True)
        payload = buffer.getvalue()
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Unique per writer, then renamed atomically: two processes rendering the
            # same image must never share a temporary path or they interleave bytes.
            tmp = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.{id(payload):x}.tmp")
            tmp.write_bytes(payload)
            os.replace(tmp, cache_path)
    return "data:image/png;base64," + base64.b64encode(payload).decode()


# Each image costs about 256 prompt tokens. No request may claim more than this many
# images, so a pathological study cannot silently blow the 128k window: the cap is a
# hard safety rail, well below the point where the request would fail outright.
MAX_IMAGES_HARD_CAP = 256
TOKENS_PER_IMAGE = 256


@dataclass(frozen=True)
class ImageSpec:
    """How a study is turned into pictures. This is an experimental axis in its own right.

    Series run to 320 slices and studies to 589 slices in this archive, so sending
    everything would cost ~150k tokens for the largest study and exceed the context
    window for about 2% of them. Every field below is a different answer to that.
    """

    plan: str = "tri_fluid_plus_t1"
    slices_per_series: int = 4
    size: int = 896
    window: str = "percentile"
    layout: str = "individual"  # or "montage" (one tiled image per series)
    sampling: str = "fixed"  # fixed | proportional | slab_mip | slab_mean
    stride: int = 8  # proportional sampling: take roughly every stride-th slice
    max_images: int = 32  # study-level ceiling; images are shared out across series
    center_fraction: float = 0.7
    # Test-time augmentation. `phase` shifts which slices are sampled; `flip` mirrors
    # the image left-right. Note that a left-right flip is not anatomically neutral
    # here: on coronal and axial images it swaps medial for lateral, and on sagittal
    # it swaps anterior for posterior, so it is expected to damage exactly the
    # laterality-dependent findings. That is why it is measured per finding.
    phase: float = 0.0
    flip: bool = False

    def key(self) -> str:
        suffix = f"-ph{self.phase:g}" if self.phase else ""
        suffix += "-flip" if self.flip else ""
        return (
            f"{self.plan}-s{self.slices_per_series}-p{self.size}"
            f"-{self.window}-{self.layout}-{self.sampling}"
            f"-st{self.stride}-m{self.max_images}-c{self.center_fraction}{suffix}"
        )

    def budget(self) -> int:
        return min(self.max_images or MAX_IMAGES_HARD_CAP, MAX_IMAGES_HARD_CAP)


def allocate_budget(wanted: list[int], budget: int) -> list[int]:
    """Share a study-level image budget across series that together want more.

    Trimming is round-robin from the largest request down, so a study with one long
    series and several short ones loses slices from the long one first instead of
    dropping short series entirely.
    """
    allocated = list(wanted)
    while sum(allocated) > budget:
        reducible = [i for i, count in enumerate(allocated) if count > 1]
        if not reducible:
            break
        largest = max(reducible, key=lambda i: allocated[i])
        allocated[largest] -= 1
    # Still over budget only when every series is down to a single image.
    if sum(allocated) > budget:
        allocated = [1] * budget
    return allocated


def plan_study_images(
    data_dir: Path, series: pd.DataFrame, uid: str, spec: ImageSpec
) -> tuple[list[tuple[SeriesRef, tuple[str, ...]]], list[int]]:
    """Decide which series contribute how many images, without decoding any pixels.

    Series are selected first, then each is asked how many images it wants under the
    sampling strategy, then the study-level budget trims the total. A montage counts
    as one image however many slices it tiles, which is what makes it a way to buy
    coverage cheaply in tokens. Splitting this out lets the runner report the token
    cost of a condition before spending anything on it.
    """
    root = data_dir / "train_series"
    stacks: list[tuple[SeriesRef, tuple[str, ...]]] = []
    for ref in select_series(series, uid, spec.plan):
        paths = ordered_slice_paths(str(root / uid / ref.series_uid))
        if paths:
            stacks.append((ref, paths))
    if not stacks:
        return [], []

    wanted = [planned_take(len(paths), spec) for _, paths in stacks]
    # A montage collapses its series into a single image, so the budget is spent in
    # units of montages rather than slices.
    budget_units = spec.budget()
    if spec.layout == "montage":
        stacks = stacks[:budget_units]
        allocated = wanted[: len(stacks)]
    else:
        allocated = allocate_budget(wanted, budget_units)
    return stacks, allocated


def image_count(spec: ImageSpec, allocated: list[int]) -> int:
    """Images a study will send: one per montage, otherwise one per sampled slice."""
    return len(allocated) if spec.layout == "montage" else sum(allocated)


def slices_represented(
    spec: ImageSpec, stacks: list[tuple[SeriesRef, tuple[str, ...]]], allocated: list[int]
) -> int:
    """How many acquired slices actually reach the model, however they are packed.

    This is the coverage side of the trade that `image_count` prices. A montage and a
    slab projection both let many slices reach the model for one image's worth of
    tokens, at the cost of per-slice resolution or of collapsing them together.
    """
    total = 0
    for (_, paths), n_take in zip(stacks, allocated):
        if spec.sampling in ("slab_mip", "slab_mean"):
            total += sum(
                upper - lower
                for lower, upper in slab_bounds(len(paths), n_take, spec.center_fraction)
            )
        else:
            total += min(n_take, len(paths))
    return total


def build_study_images(
    data_dir: Path,
    series: pd.DataFrame,
    uid: str,
    spec: ImageSpec,
    cache_dir: Path | None = None,
) -> list[tuple[str, str]]:
    """Return [(caption, data_uri)] for one study under the given image spec."""
    stacks, allocated = plan_study_images(data_dir, series, uid, spec)
    output: list[tuple[str, str]] = []
    for (ref, paths), n_take in zip(stacks, allocated):
        if n_take < 1:
            continue
        n_slices = len(paths)

        # Each renderer is a thunk: png_data_uri only calls it on a cache miss, so a
        # warm cache costs a file read rather than a DICOM decode.
        if spec.sampling in ("slab_mip", "slab_mean"):
            bounds = slab_bounds(n_slices, n_take, spec.center_fraction)
            renderers = [
                partial(
                    reduce_slab,
                    list(paths[lower:upper]),
                    spec.size,
                    spec.window,
                    spec.sampling,
                    spec.flip,
                )
                for lower, upper in bounds
            ]
            tags = [f"{lower}-{upper}" for lower, upper in bounds]
            kind = "maximum-intensity projection" if spec.sampling == "slab_mip" else "average"
            detail = f"each image is the {kind} of a contiguous slab of slices"
        else:
            indices = sample_indices(n_slices, n_take, spec.center_fraction, spec.phase)
            renderers = [
                partial(render_slice, paths[i], spec.size, spec.window, spec.flip)
                for i in indices
            ]
            tags = [str(i) for i in indices]
            detail = f"sampled from a {n_slices}-slice acquisition"

        if spec.layout == "montage":
            cache = (
                cache_dir / uid / f"{ref.series_uid}_montage_{spec.key()}.png"
                if cache_dir
                else None
            )
            caption = (
                f"{ref.caption()}; {len(renderers)} slices tiled left-to-right, "
                f"top-to-bottom, ordered through the joint, {detail}"
            )
            # A montage is one cache entry, so its tiles are rendered only together.
            output.append(
                (
                    caption,
                    png_data_uri(
                        lambda: montage([make() for make in renderers], spec.size), cache
                    ),
                )
            )
        else:
            for position, (tag, make_image) in enumerate(zip(tags, renderers), start=1):
                cache = (
                    cache_dir / uid / f"{ref.series_uid}_{tag}_{spec.key()}.png"
                    if cache_dir
                    else None
                )
                caption = (
                    f"{ref.caption()}; image {position} of {len(renderers)}, {detail}"
                )
                output.append((caption, png_data_uri(make_image, cache)))

    return output
