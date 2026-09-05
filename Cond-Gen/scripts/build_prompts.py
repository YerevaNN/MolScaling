#!/usr/bin/env python3
"""Build shared Cond-Gen prompt JSONL files into Cond-Gen/prompts (in-repo)."""
from __future__ import annotations

import argparse
from pathlib import Path

from cond_gen_utils import (
    ALL_TASKS,
    PREDEFINED_TASKS,
    default_prompts_dir,
    quiet_rdkit,
    write_task_prompts,
)


def main() -> None:
    quiet_rdkit()
    ap = argparse.ArgumentParser(description="Write shared Cond-Gen prompts (not into result dirs)")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--tasks", nargs="+", default=list(PREDEFINED_TASKS), choices=list(ALL_TASKS))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fewshot_k", type=int, default=0)
    ap.add_argument("--mol_tag_style", type=str, default="chemllama", choices=["chemllama", "chemlactica"])
    ap.add_argument("--zinc_csv", type=str, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else default_prompts_dir()
    zinc = Path(args.zinc_csv) if args.zinc_csv else None
    manifest = write_task_prompts(
        out_dir=out_dir,
        tasks=args.tasks,
        n=args.n,
        seed=args.seed,
        fewshot_k=args.fewshot_k,
        mol_tag_style=args.mol_tag_style,
        zinc_csv=zinc,
    )
    print(f"Wrote {len(args.tasks)} prompt files to {out_dir}")
    print(manifest)


if __name__ == "__main__":
    main()
