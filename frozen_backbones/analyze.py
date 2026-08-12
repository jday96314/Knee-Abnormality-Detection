#!/usr/bin/env python3
"""Summarize the sweep: which axis actually moves macro AUC, and by how much.

A ranked list of 480 configurations mostly shows noise at the top. What the
experiment is for is the marginal question -- does this backbone beat that one,
does plane-conditioning pay, does max-pooling beat mean -- so each axis is
reported as a distribution over all the configurations that share its level,
not as the single best row containing it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import data

RESULTS = Path(__file__).resolve().parent / "results"


def axis_table(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = frame.groupby(column)["macro_auc"]
    return pd.DataFrame({
        "n": grouped.size(),
        "mean": grouped.mean(),
        "median": grouped.median(),
        "best": grouped.max(),
    }).sort_values("median", ascending=False)


def derived_axes(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    bits = frame["pooling"].str.split("-")
    frame["reducer"] = bits.str[0]
    frame["grouping"] = bits.str[1]
    frame["inner"] = bits.apply(
        lambda b: "mean" if "bal" in b
        else next((x[2:] for x in b if x.startswith("in")), "none"))
    frame["across"] = bits.apply(
        lambda b: next((x[2:] for x in b if x.startswith("ax")), "concat"))
    frame["central"] = bits.apply(lambda b: "ctr" in b)
    frame["slice_l2"] = bits.apply(lambda b: "l2" in b)
    return frame


def main(csv: str = "sweep.csv") -> None:
    frame = derived_axes(pd.read_csv(RESULTS / csv))
    pd.set_option("display.width", 200)

    print(f"{len(frame)} configurations, macro AUC "
          f"{frame.macro_auc.min():.4f} - {frame.macro_auc.max():.4f}\n")

    for axis in ["backbone", "size", "head", "reducer", "grouping",
                 "inner", "across", "central", "slice_l2"]:
        print(f"-- {axis} " + "-" * (58 - len(axis)))
        print(axis_table(frame, axis).to_string(float_format=lambda v: f"{v:.4f}"))
        print()

    print("-- best configuration per backbone " + "-" * 25)
    best = frame.loc[frame.groupby("backbone")["macro_auc"].idxmax()]
    print(best[["backbone", "size", "head", "pooling", "dim", "macro_auc"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()

    print("-- top 15 overall " + "-" * 42)
    print(frame.nlargest(15, "macro_auc")[
        ["backbone", "size", "head", "pooling", "dim", "macro_auc"]]
        .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()

    print("-- per-label AUC of the single best configuration " + "-" * 10)
    top = frame.nlargest(1, "macro_auc").iloc[0]
    studies, _ = data.cohort()
    per_label = pd.DataFrame({
        "positives": [int(studies[l].sum()) for l in data.LABELS],
        "auc": [top[f"auc/{l}"] for l in data.LABELS],
    }, index=data.LABELS).sort_values("auc", ascending=False)
    print(per_label.to_string(float_format=lambda v: f"{v:.4f}"))
    # top["head"], not top.head -- attribute access finds DataFrame.head().
    print(f"\nmacro {top['macro_auc']:.4f}  "
          f"({top['backbone']}/{top['size']}/{top['head']}/{top['pooling']})")

    # How often does a label beat chance across every configuration? A label
    # that is above 0.5 in almost every run is carrying real signal; one that
    # straddles 0.5 is being picked up by chance in whichever run tops the list.
    print("\n-- per-label AUC across all configurations " + "-" * 17)
    rows = []
    for label in data.LABELS:
        column = frame[f"auc/{label}"]
        rows.append({
            "label": label,
            "positives": int(studies[label].sum()),
            "median": column.median(),
            "p10": column.quantile(0.10),
            "p90": column.quantile(0.90),
            "frac>0.5": float((column > 0.5).mean()),
        })
    print(pd.DataFrame(rows).sort_values("median", ascending=False)
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "sweep.csv")
