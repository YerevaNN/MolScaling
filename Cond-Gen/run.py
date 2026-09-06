#!/usr/bin/env python3
"""MolScaling Cond-Gen entrypoint.

  python Cond-Gen/run.py --model chemllama-170m --task qed
  python Cond-Gen/run.py --model all --tasks all
  python Cond-Gen/run.py --models chemllama-170m chemllama-380m --tasks qed sas
  python Cond-Gen/run.py --plot
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cond_gen import (
    ROOT,
    load_config,
    load_model,
    quiet_rdkit,
    resolve_model,
    run_task,
    write_scaling_plots,
)


def _unique(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def expand_models(cfg: dict, values: list[str]) -> list[str]:
    aliases = list(cfg["models"])
    out: list[str] = []
    for v in values:
        if v == "all":
            out.extend(aliases)
        else:
            out.append(v)
    return _unique(out)


def expand_tasks(cfg: dict, values: list[str]) -> list[str]:
    known = list(cfg["tasks"])
    out: list[str] = []
    for v in values:
        if v == "all":
            out.extend(known)
        else:
            out.append(v)
    unknown = [t for t in out if t not in cfg["tasks"]]
    if unknown:
        raise SystemExit(f"Unknown tasks {unknown}. Known: {known + ['all']}")
    return _unique(out)


def main() -> None:
    quiet_rdkit()
    ap = argparse.ArgumentParser(description="Conditional molecule generation (MolScaling Cond-Gen)")
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument(
        "--model",
        action="append",
        default=[],
        help="Alias from config.yaml, Hub id, local path, or 'all'. Repeatable.",
    )
    ap.add_argument("--models", nargs="+", default=[], help="One or more aliases, or 'all'")
    ap.add_argument("--task", action="append", default=[], help="Task name or 'all'. Repeatable.")
    ap.add_argument("--tasks", nargs="+", default=[], help="One or more tasks, or 'all'")
    ap.add_argument("--plot", action="store_true", help="Only write scaling figures from existing summaries")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    results_dir = ROOT / cfg["io"]["results_dir"]

    if args.plot:
        write_scaling_plots(cfg, results_dir, results_dir / "plots")
        return

    models = expand_models(cfg, list(args.model) + list(args.models))
    if not models:
        raise SystemExit("Pass --model / --models (use 'all' for every alias in config.yaml), or --plot")

    task_args = list(args.task) + list(args.tasks)
    tasks = expand_tasks(cfg, task_args if task_args else ["all"])

    gen = cfg["generation"]
    for spec in models:
        alias, model_id = resolve_model(cfg, spec)
        print(f"model={alias} id={model_id} tasks={tasks}")
        model, tokenizer = load_model(
            model_id,
            device=gen["device"],
            dtype=gen["dtype"],
            attn=gen.get("attn_implementation", "sdpa"),
        )
        for task in tasks:
            print(f"task={task}")
            s = run_task(
                cfg=cfg,
                model=model,
                tokenizer=tokenizer,
                model_alias=alias,
                model_id=model_id,
                task=task,
                overwrite=args.overwrite,
            )
            print(json.dumps({k: s[k] for k in ("property", "valid_rate", "mae", "per_property")}))
        del model, tokenizer
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    write_scaling_plots(cfg, results_dir, results_dir / "plots")


if __name__ == "__main__":
    main()
