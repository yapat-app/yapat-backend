"""
Generate paper figures from benchmarks/results.csv.

Produces a 4-panel figure
and benchmarks/figures/scalability.png.

Usage:
    python -m benchmarks.plot
    python -m benchmarks.plot --csv benchmarks/results.csv --out benchmarks/figures/scalability
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import csv


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Maps operation names in results.csv to panel labels
_PANELS = {
    "embedding":          ("(a) Embedding generation",          "GPU / CPU",  "BirdNET snippets"),
    "inference":          ("(b) AL inference (forward pass)",   "GPU / CPU",  "Snippets"),
    "retrain":            ("(c) Retraining (linear head)",      "CPU",        "Labeled snippets"),
    "similarity_search":  ("(d) pgvector cosine search",        "CPU",        "Snippets in DB"),
}

# Device display names
_DEVICE_LABELS = {
    "cpu":  "CPU",
    "cuda": "GPU (RTX PRO 6000)",
}

_COLORS = {
    "cpu":  "#2196F3",
    "cuda": "#F44336",
}

_MARKER = {
    "cpu":  "o",
    "cuda": "s",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def group_rows(rows: list[dict]) -> dict:
    """
    Returns {operation: {device: [(N, time_s_or_mean), ...]}} sorted by N.
    Handles both stage-timer rows (time_s) and runner rows (time_mean_s / time_std_s).
    """
    data: dict = {}
    for row in rows:
        op = row.get("operation", "").strip()
        if op not in _PANELS:
            continue
        device = row.get("device", "cpu").strip()
        N = _to_float(row.get("N"))
        if N is None:
            continue
        # Prefer time_mean_s if present, fall back to time_s
        t = _to_float(row.get("time_mean_s")) or _to_float(row.get("time_s"))
        std = _to_float(row.get("time_std_s")) or 0.0
        if t is None:
            continue
        data.setdefault(op, {}).setdefault(device, []).append((int(N), t, std))

    # Sort each device series by N
    for op in data:
        for dev in data[op]:
            data[op][dev].sort(key=lambda x: x[0])

    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_panel(ax, op: str, device_series: dict):
    title, device_label, xlabel = _PANELS[op]

    for device, points in device_series.items():
        Ns = [p[0] for p in points]
        ts = [p[1] for p in points]
        stds = [p[2] for p in points]
        label = _DEVICE_LABELS.get(device, device)
        color = _COLORS.get(device, "#888888")
        marker = _MARKER.get(device, "^")

        ax.plot(Ns, ts, marker=marker, color=color, label=label, linewidth=1.8, markersize=5)
        if any(s > 0 for s in stds):
            lo = [t - s for t, s in zip(ts, stds)]
            hi = [t + s for t, s in zip(ts, stds)]
            ax.fill_between(Ns, lo, hi, alpha=0.15, color=color)

    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Wall-clock time (s)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    if device_series:
        ax.legend(fontsize=7, framealpha=0.8)


def make_figure(data: dict, out_stem: str):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    fig.suptitle("YAPAT pipeline scalability", fontsize=11, fontweight="bold", y=1.01)

    panel_order = ["embedding", "similarity_search", "inference", "retrain"]
    for ax, op in zip(axes.flat, panel_order):
        plot_panel(ax, op, data.get(op, {}))

    fig.tight_layout()

    os.makedirs(os.path.dirname(out_stem) or ".", exist_ok=True)
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_stem}.png", dpi=150, bbox_inches="tight")
    print(f"Saved {out_stem}.pdf and {out_stem}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_csv():
    return os.path.join(os.path.dirname(__file__), "results.csv")


def _default_out():
    return os.path.join(os.path.dirname(__file__), "figures", "scalability")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=_default_csv())
    p.add_argument("--out", default=_default_out(),
                   help="Output file stem (no extension)")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print(f"No results file found at {args.csv}")
        return

    rows = load_csv(args.csv)
    data = group_rows(rows)
    print(f"Loaded {len(rows)} rows, operations: {list(data.keys())}")
    make_figure(data, args.out)


if __name__ == "__main__":
    main()
