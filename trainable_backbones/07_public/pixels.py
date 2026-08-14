#!/usr/bin/env python3
"""The public notebooks' pixel pipeline, ported to the local corpus.

The community notebooks in `planning/public_notebooks` do not read a study the way the
rest of this repository does. Everything here reads whole ordered stacks and pools them;
they read a *slot* — one series per (plane x acquisition) combination, three physically
spread slices out of it, cropped to a constant number of millimetres and mirrored onto a
left-knee convention. That is a different dataset over the same DICOMs, and it is the
half of those notebooks worth measuring separately from the encoder that sits on top.

Ported rather than imported, because the notebooks are inference scripts for a platform
that mounts its data elsewhere and packages weights this repository does not have. The
decisions that decide the pixels are reproduced exactly:

    order          slices sorted by position projected on the slice normal
    lat            side from the Laterality tag, else from the image centre in patient x
    slot           one series per (plane, fluid-sensitivity, fat-suppression), most slices wins
    band           samples spread over the central 20-80% of the stack
    crop           130 mm of anatomy, then resize — not a fixed pixel box
    intensity      per-series 1st-99th percentile

`RULES_LEGACY` is kept because the notebooks keep it: an imported ensemble member was
fitted under a different reading of the same four decisions, and a member read under the
wrong one produces plausible predictions from the wrong pixels. Nothing here imports such
a member, so the legacy rules exist only as a variant the sweep can ask for.

    python pixels.py --img 336 --slices 9        # build the cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data" / "from_host"
CACHE = Path(os.environ.get("KNEE_SLOT_CACHE", HERE / "cache"))

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

# --- the notebook's constants, unchanged ------------------------------------ #

CROP_MM = 130.0
SLICE_BAND = (0.20, 0.80)
LAT_MIN_OFFSET_MM = 20.0
LEGACY_LAT_OFFSET_MM = 5.0
HDR_THREADS = 24
ORDER_THREADS = 32
PIX_THREADS = 16

RULES_NATIVE = {"order": "normal", "lat": "centre",
                "slot_fallback": False, "decode_fill": "nearest"}
RULES_LEGACY = {"order": "dominant_axis", "lat": "corner_x",
                "slot_fallback": True, "decode_fill": "zero"}

# Six slots: three planes crossed with the acquisition axes. The delivered
# `Fluid_Sensitive` and `Fat_Suppression` columns of train_series.csv are byte-identical
# on this corpus — one flag published twice — so the two axes are recovered from the
# DICOM headers instead, which is what `annotate` below is for.
SLOTS_RECOVERED = [
    ("SAG_FLUID_FS", "Sagittal", True, True),
    ("COR_FLUID_FS", "Coronal", True, True),
    ("AX_FLUID_FS", "Axial", True, True),
    ("SAG_FLUID_NOFS", "Sagittal", True, False),
    ("COR_T1", "Coronal", False, False),
    ("SAG_T1", "Sagittal", False, False),
]
SLOTS_PUBLIC = [
    ("SAG_FLUID", "Sagittal", None, True),
    ("COR_FLUID", "Coronal", None, True),
    ("AX_FLUID", "Axial", None, True),
    ("SAG_STRUCT", "Sagittal", None, False),
    ("COR_STRUCT", "Coronal", None, False),
    ("AX_STRUCT", "Axial", None, False),
]
# The RadImageNet arm reads three fat-suppressed slots at full frame instead.
SLOTS_RAD = [
    ("SAG_FS", "Sagittal", None, True),
    ("COR_FS", "Coronal", None, True),
    ("AX_FS", "Axial", None, True),
]
SCHEMES = {"recovered": SLOTS_RECOVERED, "public": SLOTS_PUBLIC, "rad": SLOTS_RAD}

FATSAT_OPTS = {"FS", "FATSAT", "FAT_SAT", "FSAT"}
_SEP = re.compile(r"[_\-.]")
_FATSAT_RX = re.compile(r"\bfs\b|fatsat|fat sat|\bstir\b|\bspair\b|\bspir\b|\bwe\b|"
                        r"water excit|\btirm\b|\bsting\b|\bfatsup\b")
_T1_RX = re.compile(r"\bt1\b|\bt1w\b")
_T2_RX = re.compile(r"\bt2\b|\bt2w\b")
_PD_RX = re.compile(r"\bpd\b|\bpdw\b|proton|\bdp\b|dens")

HDR_TAGS = ["SeriesDescription", "SequenceName", "ScanOptions", "ScanningSequence",
            "RepetitionTime", "EchoTime", "Laterality", "PixelSpacing", "Rows",
            "Columns", "RescaleSlope", "RescaleIntercept",
            "ImagePositionPatient", "ImageOrientationPatient"]
ORDER_TAGS = [(0x0020, 0x0032), (0x0020, 0x0037), (0x0020, 0x0013)]

T0 = time.time()
DECODE_FAILED: list[str] = []


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# --- header pass ------------------------------------------------------------ #


def probe(item):
    study, series, path = item
    row = {"StudyInstanceUID": study, "SeriesInstanceUID": series, "dir": str(path)}
    try:
        files = sorted(e.name for e in os.scandir(path) if e.name.endswith(".dcm"))
        row["files"] = files
        row["n_slices"] = len(files)
        if not files:
            return row
        ds = pydicom.dcmread(os.path.join(path, files[len(files) // 2]),
                             stop_before_pixels=True, force=True)
        for t in HDR_TAGS:
            v = getattr(ds, t, None)
            if v is None:
                row[t] = None
            elif isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
                row[t] = "|".join(str(x) for x in v)
            else:
                row[t] = str(v)
    except Exception as exc:  # noqa: BLE001 - a bad header must not stop the pass
        row["err"] = str(exc)[:120]
    return row


def walk(split="train_series"):
    """One header read per series of a split."""
    base = DATA / split
    items = [(study.name, series.name, series.path)
             for study in os.scandir(base) if study.is_dir()
             for series in os.scandir(study.path) if series.is_dir()]
    with ThreadPoolExecutor(max_workers=HDR_THREADS) as pool:
        rows = list(pool.map(probe, items))
    return pd.DataFrame(rows)


def annotate(df):
    """Recover fat suppression and pulse-sequence weighting from the header."""
    desc = (df["SeriesDescription"].fillna("") + " " + df["SequenceName"].fillna(""))
    desc = desc.str.lower().str.replace(_SEP, " ", regex=True)

    opts = df["ScanOptions"].fillna("").str.upper().str.split("|")
    # GE writes SAT_GEMS for spatial saturation, so ScanOptions is matched as exact
    # tokens; a substring test on "SAT" fires on series that are not fat-suppressed.
    opts_fs = opts.apply(lambda ts: any(t.strip() in FATSAT_OPTS for t in ts))
    df["fatsat"] = desc.str.contains(_FATSAT_RX) | opts_fs

    tr = pd.to_numeric(df["RepetitionTime"], errors="coerce")
    te = pd.to_numeric(df["EchoTime"], errors="coerce")
    gre = df["ScanningSequence"].fillna("").str.upper().str.contains("GR")
    t1, t2, pdw = desc.str.contains(_T1_RX), desc.str.contains(_T2_RX), desc.str.contains(_PD_RX)

    df["weight"] = np.where(t1 & ~t2 & ~pdw, "T1",
                     np.where(t2 & ~pdw, "T2",
                       np.where(pdw, "PD",
                         np.where(gre, "GRE",
                           np.where(tr < 800, "T1",
                             np.where(te > 60, "T2",
                               np.where(tr >= 800, "PD", "UNK")))))))
    df["fluid"] = np.isin(df["weight"], ["PD", "T2"])
    df["px"] = pd.to_numeric(
        df["PixelSpacing"].fillna("").str.split("|").str[0].replace("", np.nan),
        errors="coerce")
    return df


# --- laterality ------------------------------------------------------------- #


def _hdr_vec(s, n):
    if not isinstance(s, str):
        return None
    try:
        v = [float(x) for x in s.split("|")]
    except ValueError:
        return None
    return np.array(v) if len(v) >= n else None


def side_from_geometry(h):
    """Study -> 'L'/'R'/None from where the image centre sits in patient x.

    `Laterality` is absent on half this corpus, and absent by vendor rather than at
    random, so treating an untagged study as left-sided leaves half the corpus
    unmirrored — with the five side-defined findings presented on a reversed axis.
    """
    cx = {}
    for r in h.itertuples(index=False):
        ipp = _hdr_vec(getattr(r, "ImagePositionPatient", None), 3)
        iop = _hdr_vec(getattr(r, "ImageOrientationPatient", None), 6)
        ps = _hdr_vec(getattr(r, "PixelSpacing", None), 2)
        rows, cols = getattr(r, "Rows", None), getattr(r, "Columns", None)
        if ipp is None or iop is None or ps is None or not rows or not cols:
            continue
        try:
            c = ipp[:3] + iop[:3] * ps[1] * float(cols) / 2 + iop[3:6] * ps[0] * float(rows) / 2
        except (TypeError, ValueError):
            continue
        cx.setdefault(r.StudyInstanceUID, []).append(float(c[0]))
    out = {}
    for st, xs in cx.items():
        m = float(np.median(xs))
        out[st] = None if abs(m) < LAT_MIN_OFFSET_MM else ("R" if m < 0 else "L")
    return out


def side_from_corner_x(h):
    """The legacy rule: the raw corner x, with a 5 mm dead zone."""
    out = {}
    for st, g in h.groupby("StudyInstanceUID"):
        xs = []
        for r in g.itertuples(index=False):
            ipp = _hdr_vec(getattr(r, "ImagePositionPatient", None), 3)
            if ipp is not None and np.isfinite(ipp).all():
                xs.append(float(ipp[0]))
        if not xs:
            out[st] = None
            continue
        x = float(np.median(xs))
        out[st] = None if abs(x) < LEGACY_LAT_OFFSET_MM else ("R" if x < 0 else "L")
    return out


def lat_of(h, rules, tag=""):
    geo = side_from_corner_x(h) if rules["lat"] == "corner_x" else side_from_geometry(h)
    d, n_tag, n_geo, n_none, n_disagree = {}, 0, 0, 0, 0
    for st, g in h.groupby("StudyInstanceUID"):
        v = [str(x).strip().upper() for x in g["Laterality"].dropna()]
        v = [x[0] for x in v if x and x[0] in ("L", "R")]
        side = v[0] if v else None
        if side is not None:
            n_tag += 1
            if geo.get(st) is not None and geo[st] != side:
                n_disagree += 1
        else:
            side = geo.get(st)
            n_geo += side is not None
            n_none += side is None
        d[st] = side
    log(f"{tag}laterality: {n_tag} from the tag, {n_geo} from geometry, {n_none} "
        f"unresolved; tag and geometry disagree on {n_disagree} "
        f"({n_disagree / max(n_tag, 1):.1%} of the tagged)")
    return d


def normalise_laterality(img, plane, lat):
    """Map every knee onto a left-knee convention.

    Coronal and axial views mirror under a horizontal flip. Sagittal stacks are not
    mirror images of each other — the slice order runs medial-to-lateral in opposite
    directions — so the channel order is reversed instead.
    """
    if lat != "R":
        return img
    if plane in ("Coronal", "Axial"):
        return torch.flip(img, dims=[-1])
    return torch.flip(img, dims=[0])


# --- slot selection --------------------------------------------------------- #


def pick_slots(series_df, plane_map, slots, rules):
    """One series per slot per study; ties go to the stack with the most slices."""
    series_df = series_df.copy()
    series_df["plane"] = series_df["SeriesInstanceUID"].map(plane_map)
    out = {}
    for study, g in series_df.groupby("StudyInstanceUID"):
        chosen = {}
        for name, plane, fluid, fs in slots:
            sel = (g["plane"] == plane) & (g["fatsat"] == fs)
            if fluid is not None:
                sel &= (g["fluid"] == fluid)
            cand = g[sel]
            if len(cand) == 0 and rules["slot_fallback"] and fluid is False:
                cand = g[(g["plane"] == plane) & (~g["fatsat"])]
            if len(cand):
                # A plain dict, not the DataFrame row: the ordering pass writes an
                # `ordered` key back onto it, and assigning a list into a Series item is
                # an alignment operation rather than a store.
                chosen[name] = cand.sort_values(
                    "n_slices", ascending=False).iloc[0].to_dict()
        out[study] = chosen
    return out


# --- slice ordering --------------------------------------------------------- #


def _natural_key(name):
    return tuple(int(x) if x.isdigit() else x.lower()
                 for x in re.split(r"(\d+)", str(name)))


def _order_dominant_axis(rec):
    files, d = rec["files"], rec["dir"]
    rows = []
    for pos, f in enumerate(files):
        ipp = inst = None
        try:
            ds = pydicom.dcmread(os.path.join(d, f), force=True, stop_before_pixels=True,
                                 specific_tags=["ImagePositionPatient", "InstanceNumber"])
            raw = getattr(ds, "ImagePositionPatient", None)
            if raw is not None and len(raw) >= 3:
                c = np.asarray(raw[:3], dtype=np.float64)
                if np.isfinite(c).all():
                    ipp = c
            n = getattr(ds, "InstanceNumber", None)
            if n is not None:
                inst = float(n)
        except Exception:  # noqa: BLE001
            pass
        rows.append((f, ipp, inst, pos))

    placed = [r for r in rows if r[1] is not None]
    need = max(2, int(0.8 * len(rows)))
    if len(placed) >= need:
        xyz = np.stack([r[1] for r in placed])
        axis = int(np.argmax(np.ptp(xyz, axis=0)))
        spare = float(np.nanmedian(xyz[:, axis]))
        rows.sort(key=lambda r: (float(r[1][axis]) if r[1] is not None else spare,
                                 r[2] if r[2] is not None else float("inf"), r[3]))
    elif sum(r[2] is not None for r in rows) >= need:
        rows.sort(key=lambda r: (r[2] if r[2] is not None else float("inf"), r[3]))
    else:
        rows.sort(key=lambda r: _natural_key(r[0]))
    return [r[0] for r in rows], True


def order_slices(rec, rules):
    """Files sorted along the through-plane axis, k = p . (r_x x r_y).

    A DICOM file name here is a SOP Instance UID, assigned arbitrarily, so file order is
    uncorrelated with anatomy. Anything that assumes otherwise is reading noise: the
    three channels of a 2.5D input become three unrelated views, "the middle of the
    stack" becomes a random subset, and reversing slice order to mirror a right knee
    reverses nothing.
    """
    if rules["order"] == "dominant_axis":
        return _order_dominant_axis(rec)
    files, d = rec["files"], rec["dir"]
    keyed = []
    for f in files:
        k = None
        try:
            ds = pydicom.dcmread(os.path.join(d, f), force=True, stop_before_pixels=True,
                                 specific_tags=ORDER_TAGS)
            iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
            ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
            k = float(np.dot(ipp, np.cross(iop[:3], iop[3:])))
        except Exception:  # noqa: BLE001
            try:
                k = float(ds.InstanceNumber)
            except Exception:  # noqa: BLE001
                k = None
        keyed.append((k, f))
    if any(k is None for k, _ in keyed):
        return files, False
    return [f for _, f in sorted(keyed, key=lambda t: t[0])], True


# --- pixels ----------------------------------------------------------------- #


def read_slot(rec, n_slice, out_size, crop_mm, band, rules):
    """`n_slice` physically spread slices from one series, at `out_size` pixels.

    Percentile normalisation rather than min/max because MR intensity has no absolute
    scale and one bright vessel would otherwise compress the joint into a narrow band.
    """
    files, d, px = rec.get("ordered") or rec["files"], rec["dir"], rec["px"]
    n = len(files)
    if n == 0:
        return None
    lo, hi = int(band[0] * (n - 1)), int(band[1] * (n - 1))
    idx = np.unique(np.linspace(lo, hi, n_slice).astype(int)) if hi > lo else np.array([n // 2])
    while len(idx) < n_slice:
        idx = np.append(idx, idx[-1])

    planes = []
    for i in idx[:n_slice]:
        try:
            ds = pydicom.dcmread(os.path.join(d, files[int(i)]), force=True)
            a = ds.pixel_array.astype(np.float32)
            if a.ndim == 3:  # rare multi-frame or colour slice
                a = a[a.shape[0] // 2] if a.shape[-1] > 4 else a[..., 0]
            sl = float(getattr(ds, "RescaleSlope", 1) or 1)
            ic = float(getattr(ds, "RescaleIntercept", 0) or 0)
            a = a * sl + ic
        except Exception:  # noqa: BLE001 - no shape is known here; see below
            a = None
        planes.append(a)

    # A slice that would not decode has no shape of its own. Inventing one at the resize
    # target makes the shape check below take the substitute as the authority and zero
    # the slices that did decode, leaving a black slot the presence mask still reports as
    # acquired. Filling from the nearest slice that did decode keeps the slot honest.
    got = [k for k, p in enumerate(planes) if p is not None]
    if rules["decode_fill"] == "zero":
        if not got:
            DECODE_FAILED.append(rec["SeriesInstanceUID"])
        planes = [np.zeros((out_size, out_size), np.float32) if p is None else p
                  for p in planes]
        got = list(range(len(planes)))
    if not got:
        DECODE_FAILED.append(rec["SeriesInstanceUID"])
        return None
    if len(got) < len(planes):
        DECODE_FAILED.append(rec["SeriesInstanceUID"])
        for k, p in enumerate(planes):
            if p is None:
                planes[k] = planes[min(got, key=lambda j: abs(j - k))]

    shp = planes[0].shape
    planes = [p if p.shape == shp else np.zeros(shp, np.float32) for p in planes]
    vol = np.stack(planes)

    # A constant physical extent, then resize: PixelSpacing varies 3.4x across the
    # corpus, so a fixed pixel box is a different amount of anatomy per study.
    if px and np.isfinite(px) and px > 0:
        want = int(round(crop_mm / px))
        h, w = shp
        if 16 < want < min(h, w):
            cy, cx = h // 2, w // 2
            half = want // 2
            vol = vol[:, max(0, cy - half):cy + half, max(0, cx - half):cx + half]

    lo_v, hi_v = np.percentile(vol, [1, 99])
    vol = np.clip((vol - lo_v) / max(hi_v - lo_v, 1e-6), 0, 1)

    t = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)
    t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return (t.squeeze(0) * 255).round().clamp(0, 255).to(torch.uint8)


# --- cache ------------------------------------------------------------------ #


def cache_tag(cfg):
    """The name a decoded cache is stored under.

    It has to name everything that decides the pixels, not only their dimensions: two
    configurations agreeing on resolution, slice count, crop and band but disagreeing on
    how a slice is chosen produce different arrays of identical shape, and a tag built
    from the dimensions alone would let the second attach to the first one's file.
    """
    t = (f"{cfg['scheme']}_{cfg['img']}px_{cfg['slices']}sl_{int(cfg['crop_mm'])}mm_"
         f"{cfg['band'][0]:.2f}-{cfg['band'][1]:.2f}")
    if cfg["rules"] != RULES_NATIVE:
        t += "_" + hashlib.md5(json.dumps(cfg["rules"], sort_keys=True).encode()).hexdigest()[:6]
    return t


def config(scheme="recovered", img=336, slices=9, crop_mm=CROP_MM, band=SLICE_BAND,
           rules=None):
    return {"scheme": scheme, "img": int(img), "slices": int(slices),
            "crop_mm": float(crop_mm), "band": tuple(band),
            "rules": dict(rules or RULES_NATIVE)}


def order_cache_path():
    return CACHE / "slice_order.json"


def build(cfg, headers=None, order_cache=True):
    """Decode every (study, slot) once into a memmapped uint8 array on disk.

    Fine-tuning revisits the same pixels every epoch, and reading them from the DICOMs
    each time would make the epoch count a function of I/O rather than of learning.
    """
    tag = cache_tag(cfg)
    CACHE.mkdir(parents=True, exist_ok=True)
    npy, meta_path = CACHE / f"{tag}.u8.npy", CACHE / f"{tag}.meta.json"
    if npy.exists() and meta_path.exists():
        log(f"{tag}: already built")
        return load(cfg)

    slots = SCHEMES[cfg["scheme"]]
    h = headers if headers is not None else annotate(walk())
    series = pd.read_csv(DATA / "train_series.csv")
    plane_map = dict(zip(series["SeriesInstanceUID"], series["Anatomical_Plane"]))
    slot_map = pick_slots(h, plane_map, slots, cfg["rules"])
    lat_map = lat_of(h, cfg["rules"], "train ")

    studies = sorted(slot_map)
    sidx = {s: i for i, s in enumerate(studies)}
    fill = pd.Series([len(v) for v in slot_map.values()])
    log(f"{tag}: slots per study mean {fill.mean():.2f} min {fill.min()} max {fill.max()}")

    jobs = [(st, k, plane, slot_map[st][name])
            for st in studies
            for k, (name, plane, _, _) in enumerate(slots)
            if name in slot_map[st]]

    # Ordering first and as its own pass: it reads one header per slice of every chosen
    # series, far more file opens than the decode that follows, and the result depends on
    # neither resolution nor slice count — so it is remembered across configurations.
    seen = {}
    op = order_cache_path()
    if order_cache and op.is_file():
        try:
            seen = json.loads(op.read_text())
        except (OSError, ValueError):
            seen = {}
    # Keyed by the ordering rule as well as the series: `dominant_axis` and `normal`
    # return the same files in the opposite order, and a cache that forgets which one it
    # holds hands the second rule the first one's answer.
    okey = lambda rec: f"{cfg['rules']['order']}:{rec['SeriesInstanceUID']}"  # noqa: E731
    hit = 0
    for _, _, _, rec in jobs:
        e = seen.get(okey(rec))
        if e and len(e["files"]) == len(rec["files"]):
            rec["ordered"] = e["files"]
            hit += 1
    todo = [j for j in jobs if "ordered" not in j[3]]
    log(f"{tag}: ordering {len(todo)} slot-series ({hit} remembered) — "
        f"{sum(len(j[3]['files']) for j in todo):,} slice headers")

    t_ord, ok, done = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=ORDER_THREADS) as pool:
        for c0 in range(0, len(todo), 1024):
            block = todo[c0:c0 + 1024]
            for (_, _, _, rec), (files, good) in zip(
                    block, pool.map(lambda j: order_slices(j[3], cfg["rules"]), block)):
                rec["ordered"] = files
                ok += int(good)
                done += 1
                seen[okey(rec)] = {"files": files, "good": bool(good)}
            log(f"  {tag} ordered {done}/{len(todo)} ({time.time() - t_ord:.0f}s)")
    if order_cache and done:
        tmp = op.with_suffix(".tmp")
        tmp.write_text(json.dumps(seen))
        tmp.replace(op)
    if todo:
        log(f"{tag}: {ok}/{len(todo)} ordered by geometry in {time.time() - t_ord:.0f}s")

    cache = np.lib.format.open_memmap(
        npy, mode="w+", dtype=np.uint8,
        shape=(len(studies), len(slots), cfg["slices"], cfg["img"], cfg["img"]))
    mask = np.zeros((len(studies), len(slots)), np.float32)
    log(f"{tag}: cache {cache.shape} = {cache.nbytes / 1024 ** 3:.1f} GB -> {npy}")

    n_fail_before, done, t_dec = len(DECODE_FAILED), 0, time.time()
    with ThreadPoolExecutor(max_workers=PIX_THREADS) as pool:
        for c0 in range(0, len(jobs), 512):
            block = jobs[c0:c0 + 512]
            reader = lambda j: read_slot(j[3], cfg["slices"], cfg["img"],  # noqa: E731
                                         cfg["crop_mm"], cfg["band"], cfg["rules"])
            for (st, k, plane, _), img in zip(block, pool.map(reader, block)):
                done += 1
                if img is None:
                    continue
                cache[sidx[st], k] = normalise_laterality(img, plane, lat_map.get(st)).numpy()
                mask[sidx[st], k] = 1.0
            log(f"  {tag} decoded {done}/{len(jobs)} ({time.time() - t_dec:.0f}s)")
    cache.flush()
    n_fail = len(DECODE_FAILED) - n_fail_before
    log(f"{tag}: {int(mask.sum())}/{len(jobs)} slots filled"
        + (f"; {n_fail} series had a slice that would not decode" if n_fail else ""))

    meta_path.write_text(json.dumps({
        "config": {**cfg, "band": list(cfg["band"])},
        "slots": [s[0] for s in slots],
        "studies": studies,
        "mask": mask.tolist(),
        "laterality": {k: v for k, v in lat_map.items()},
    }))
    return load(cfg)


def load(cfg, build_if_missing=True):
    """(studies, cache, mask) for one configuration, memory-mapped.

    A configuration names its own cache, so a sweep that asks for a slot scheme nobody
    has decoded yet should decode it rather than stop: the header pass and the slice
    order are already remembered, which is most of the cost.
    """
    tag = cache_tag(cfg)
    if build_if_missing and not (CACHE / f"{tag}.meta.json").exists():
        log(f"{tag}: not decoded yet; building it now")
        return build(cfg, headers=headers())
    meta = json.loads((CACHE / f"{tag}.meta.json").read_text())
    array = np.load(CACHE / f"{tag}.u8.npy", mmap_mode="r")
    return meta["studies"], array, np.asarray(meta["mask"], np.float32)


def headers_path():
    return CACHE / "headers.pkl"


def headers(rebuild=False):
    """The header pass, remembered — it is the same for every configuration."""
    p = headers_path()
    if p.exists() and not rebuild:
        return pd.read_pickle(p)
    CACHE.mkdir(parents=True, exist_ok=True)
    log("header pass over train_series")
    h = annotate(walk())
    log(f"  {len(h)} series")
    h.to_pickle(p)
    return h


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scheme", default="recovered", choices=list(SCHEMES))
    p.add_argument("--img", type=int, default=336)
    p.add_argument("--slices", type=int, default=9)
    p.add_argument("--crop-mm", type=float, default=CROP_MM)
    p.add_argument("--band", type=float, nargs=2, default=list(SLICE_BAND))
    p.add_argument("--rules", default="native", choices=["native", "legacy"])
    a = p.parse_args()
    cfg = config(a.scheme, a.img, a.slices, a.crop_mm, tuple(a.band),
                 RULES_LEGACY if a.rules == "legacy" else RULES_NATIVE)
    studies, cache, mask = build(cfg, headers=headers())
    log(f"{cache_tag(cfg)}: {len(studies)} studies, {cache.shape}, "
        f"{mask.mean():.3f} of slots filled")


if __name__ == "__main__":
    main()
