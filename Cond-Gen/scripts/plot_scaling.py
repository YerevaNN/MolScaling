#!/usr/bin/env python3
"""MAE vs model-size scaling plots for single / double / triple Cond-Gen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cond_gen_utils import DOUBLE_TASKS, SINGLE_TASKS, TRIPLE_TASKS, ensure_dir

MODEL_ORDER = ("chemllama-170m", "chemllama-380m", "chemllama-1b", "chemllama-3b")
MODEL_LABELS = {
    "chemllama-170m": "170M",
    "chemllama-380m": "380M",
    "chemllama-1b": "1.3B",
    "chemllama-3b": "3.2B",
}
# SAS is typically ~1–5; divide by 5 so it sits on a QED-like [0, 1] scale.
SAS_NORM = 5.0
PROP_STYLE = {
    "qed": {"color": "#1f77b4", "marker": "o", "ylabel": "MAE", "title": "QED"},
    "sas": {"color": "#ff7f0e", "marker": "s", "ylabel": "MAE", "title": "SA"},
    "similarity": {"color": "#2ca02c", "marker": "^", "ylabel": "MAE", "title": "Similarity"},
    "norm": {
        "color": "#9467bd",
        "marker": "D",
        "ylabel": "QED MAE + SA MAE / 5",
        "title": "Norm. avg (QED + SA/5)",
    },
}
ARITY_TASK = {
    "single": {
        "qed": "qed",
        "sas": "sas",
        "similarity": "similarity_random",
    },
    "double": {"qed": "qed_sas", "sas": "qed_sas"},
    "double_swap": {"qed": "sas_qed", "sas": "sas_qed"},
    "triple": {
        "qed": "qed_sas_similarity_random",
        "sas": "qed_sas_similarity_random",
        "similarity": "qed_sas_similarity_random",
    },
    "triple_swap": {
        "qed": "sas_qed_similarity_random",
        "sas": "sas_qed_similarity_random",
        "similarity": "sas_qed_similarity_random",
    },
}


def _load_summaries(results_root: Path) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for path in results_root.rglob("summary.json"):
        try:
            s = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        model = str(s.get("model_size") or "")
        task = str(s.get("property") or "")
        if model and task:
            out[(model, task)] = s
    return out


def _mae(summary: dict, prop: str) -> float | None:
    per = summary.get("per_property") or {}
    if prop in per and per[prop].get("mae") is not None:
        val = float(per[prop]["mae"])
        return val if val == val else None
    keys = summary.get("property_keys") or []
    if len(keys) == 1 and keys[0] == prop and summary.get("mae") is not None:
        val = float(summary["mae"])
        return val if val == val else None
    return None


def _series(summaries: dict[tuple[str, str], dict], arity: str, prop: str) -> list[float | None]:
    task = ARITY_TASK[arity][prop]
    vals: list[float | None] = []
    for model in MODEL_ORDER:
        s = summaries.get((model, task))
        vals.append(_mae(s, prop) if s else None)
    return vals


def _norm_avg_series(summaries: dict[tuple[str, str], dict], arity: str) -> list[float | None]:
    qed = _series(summaries, arity, "qed")
    sas = _series(summaries, arity, "sas")
    out: list[float | None] = []
    for q, s in zip(qed, sas, strict=True):
        if q is None or s is None:
            out.append(None)
        else:
            out.append(q + s / SAS_NORM)
    return out


def _annotate(ax, xs, ys) -> None:
    yvals = [y for y in ys if y is not None]
    if not yvals:
        return
    span = max(yvals) - min(yvals)
    pad = 0.04 * span if span > 0 else 0.004 * (abs(max(yvals)) + 1e-6)
    for x, y in zip(xs, ys, strict=True):
        if y is None:
            continue
        ax.text(x, y + pad, f"{y:.3f}", ha="center", va="bottom", fontsize=9)


def _draw_panel(ax, ys: list[float | None], *, kind: str, title: str) -> None:
    style = PROP_STYLE[kind]
    xs = list(range(len(MODEL_ORDER)))
    x_ok = [x for x, y in zip(xs, ys, strict=True) if y is not None]
    y_ok = [y for y in ys if y is not None]
    if y_ok:
        ax.plot(
            x_ok,
            y_ok,
            color=style["color"],
            marker=style["marker"],
            markersize=8,
            linewidth=1.8,
        )
        _annotate(ax, xs, ys)
        ymin, ymax = min(y_ok), max(y_ok)
        pad = 0.22 * (ymax - ymin) if ymax > ymin else 0.08 * (abs(ymax) + 1e-6)
        ax.set_ylim(ymin - 0.4 * pad, ymax + pad)
    ax.set_title(title)
    ax.set_xlabel("Model size")
    ax.set_ylabel(style["ylabel"])
    ax.set_xticks(range(len(MODEL_ORDER)), [MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.grid(True, linestyle="--", linewidth=0.8, color="0.75")
    ax.set_axisbelow(True)


def _save_fig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_single(summaries: dict, out: Path) -> None:
    panels = [
        ("qed", _series(summaries, "single", "qed"), "QED scaling"),
        ("sas", _series(summaries, "single", "sas"), "SA scaling"),
        ("norm", _norm_avg_series(summaries, "single"), "Norm. avg scaling (QED + SA/5)"),
        ("similarity", _series(summaries, "single", "similarity"), "Similarity scaling"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.4))
    for ax, (kind, ys, title) in zip(axes.ravel(), panels, strict=True):
        _draw_panel(ax, ys, kind=kind, title=title)
    _save_fig(fig, out)


def plot_double(summaries: dict, out: Path) -> None:
    rows = (("double", "QED→SA"), ("double_swap", "SA→QED"))
    cols = (("qed", "QED"), ("sas", "SA"), ("norm", "Norm. avg (QED + SA/5)"))
    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.4))
    for r, (arity, order) in enumerate(rows):
        for c, (kind, name) in enumerate(cols):
            ys = _norm_avg_series(summaries, arity) if kind == "norm" else _series(summaries, arity, kind)
            _draw_panel(axes[r][c], ys, kind=kind, title=f"{name} ({order})")
    _save_fig(fig, out)


def plot_triple(summaries: dict, out: Path) -> None:
    rows = (("triple", "QED→SA"), ("triple_swap", "SA→QED"))
    cols = (("qed", "QED"), ("sas", "SA"), ("similarity", "Similarity"))
    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.4))
    for r, (arity, order) in enumerate(rows):
        for c, (kind, name) in enumerate(cols):
            _draw_panel(
                axes[r][c],
                _series(summaries, arity, kind),
                kind=kind,
                title=f"{name} ({order}, triple)",
            )
    _save_fig(fig, out)


def write_mae_table(summaries: dict, out_csv: Path) -> None:
    rows = ["arity,task,property," + ",".join(MODEL_ORDER)]
    for arity, tasks in (
        ("single", SINGLE_TASKS),
        ("double", DOUBLE_TASKS),
        ("triple", TRIPLE_TASKS),
    ):
        for task in tasks:
            sample = next((summaries[(m, task)] for m in MODEL_ORDER if (m, task) in summaries), None)
            keys = list((sample or {}).get("property_keys") or [])
            if not keys:
                continue
            for prop in keys:
                vals = []
                for model in MODEL_ORDER:
                    s = summaries.get((model, task))
                    v = _mae(s, prop) if s else None
                    vals.append("" if v is None else f"{v:.6f}")
                rows.append(f"{arity},{task},{prop}," + ",".join(vals))
            if "qed" in keys and "sas" in keys:
                vals = []
                for model in MODEL_ORDER:
                    s = summaries.get((model, task))
                    q = _mae(s, "qed") if s else None
                    sa = _mae(s, "sas") if s else None
                    if q is None or sa is None:
                        vals.append("")
                    else:
                        vals.append(f"{q + sa / SAS_NORM:.6f}")
                rows.append(f"{arity},{task},qed_plus_sas_over_{int(SAS_NORM)}," + ",".join(vals))
        if arity == "single":
            vals = []
            for model in MODEL_ORDER:
                sq = summaries.get((model, "qed"))
                ss = summaries.get((model, "sas"))
                q = _mae(sq, "qed") if sq else None
                sa = _mae(ss, "sas") if ss else None
                if q is None or sa is None:
                    vals.append("")
                else:
                    vals.append(f"{q + sa / SAS_NORM:.6f}")
            rows.append(f"single,qed+sas,qed_plus_sas_over_{int(SAS_NORM)}," + ",".join(vals))
    out_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Cond-Gen MAE scaling vs model size")
    ap.add_argument(
        "--results_root",
        type=str,
        default="/mnt/weka/gsimonyan/results/genetic/cond-gen/hub_yerevann_n1000_seed1",
    )
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()
    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir) if args.out_dir else results_root / "plots"
    ensure_dir(out_dir)

    summaries = _load_summaries(results_root)
    if not summaries:
        raise SystemExit(f"No summary.json files under {results_root}")

    plot_single(summaries, out_dir / "scaling_single.png")
    plot_double(summaries, out_dir / "scaling_double.png")
    plot_triple(summaries, out_dir / "scaling_triple.png")
    write_mae_table(summaries, out_dir / "mae_table.csv")
    print(f"Wrote plots to {out_dir}")
    print(f"Loaded {len(summaries)} summaries")


if __name__ == "__main__":
    main()
