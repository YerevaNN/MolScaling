#!/usr/bin/env python3
"""
Cond-Gen: unified conditional-generation evaluation pipeline.

Flow:
  1) ensure prompts (reuse if matching manifest exists, else build + save)
  2) load language model
  3) generate for each task
  4) extract SMILES
  5) compute conditioned property / similarity
  6) report MAE + RMSE and plot desired vs computed
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from cond_gen_utils import (
    ALL_TASKS,
    DEFAULT_TASKS,
    GENERATION_REPORT_FIELDS,
    compute_properties_for_task,
    default_prompts_dir,
    ensure_dir,
    ensure_prompts,
    extract_smiles,
    generation_debug_from_ids,
    load_hf_model_and_tokenizer,
    mae,
    mol_tag_style_for_model,
    plot_desired_vs_computed,
    plot_error_hist,
    project_path,
    quiet_rdkit,
    rmse,
    row_targets,
    safe_jsonl_read,
    task_property_keys,
    task_uses_similarity_extraction,
    tokenize_prompts_for_generation,
    write_run_identity_artifacts,
    write_tokenization_probe,
)


MODEL_ID_MAP = {
    "chemllama-170m": "yerevann/ChemLlama-170M",
    "chemllama-380m": "yerevann/ChemLlama-380M",
    "chemllama-1b": "yerevann/ChemLlama-1B",
    "chemllama-3b": "yerevann/ChemLlama-3B",
    "chemlactica-1.3b": "yerevann/chemlactica-1.3b",
}

# Local HF export required unless HF_LOCAL_MODEL_DIR is set.
LOCAL_ONLY_MODELS = {"chemlactica-1.3b"}


def _load_model_and_tokenizer(model_size: str, device: str, dtype: str):
    model_id = MODEL_ID_MAP.get(model_size, model_size)
    checkpoint_path = os.environ.get("HF_LOCAL_MODEL_DIR", "").strip() or model_id
    tokenizer_path = os.environ.get("HF_LOCAL_TOKENIZER_DIR", "").strip() or checkpoint_path

    if model_size in LOCAL_ONLY_MODELS and not os.environ.get("HF_LOCAL_MODEL_DIR", "").strip():
        raise RuntimeError(f"{model_size} requires HF_LOCAL_MODEL_DIR for local checkpoint loading.")

    model, tokenizer = load_hf_model_and_tokenizer(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        device=device,
        dtype=dtype,
        padding_side="left",
    )
    return model, tokenizer, model_id, checkpoint_path, tokenizer_path


def _eos_token_id_for_generation(tokenizer: AutoTokenizer) -> int | None:
    try:
        eot_ids = tokenizer.encode("<|end_of_text|>", add_special_tokens=False)
        if len(eot_ids) == 1:
            return int(eot_ids[0])
    except Exception:
        pass
    return tokenizer.eos_token_id


def run_one_task(
    *,
    model,
    tokenizer,
    model_size: str,
    model_id: str,
    task: str,
    prompts_path: Path,
    report_dir: Path,
    device: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    repetition_penalty: float,
    do_sample: bool,
    seed: int,
    overwrite: bool,
) -> dict:
    ensure_dir(report_dir)
    report_csv = report_dir / "generation_report.csv"
    summary_csv = report_dir / "summary.csv"
    summary_json = report_dir / "summary.json"
    if not overwrite and summary_json.exists():
        print(f"PIPELINE: skip existing {report_dir}")
        return json.loads(summary_json.read_text(encoding="utf-8"))

    write_tokenization_probe(
        report_dir / "processing_tokenization_probe.txt",
        tokenizer,
        note=f"run_pipeline.py model_size={model_size} model_id={model_id} task={task}",
    )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if overwrite:
        if report_csv.exists():
            report_csv.unlink()
        if summary_csv.exists():
            summary_csv.unlink()

    with report_csv.open("w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=GENERATION_REPORT_FIELDS).writeheader()

    eos_token_id = _eos_token_id_for_generation(tokenizer) or tokenizer.eos_token_id
    similarity = task_uses_similarity_extraction(task)
    prop_keys = task_property_keys(task)
    # Per-property accumulators for metrics / plots.
    per_prop_true: dict[str, list[float]] = {k: [] for k in prop_keys}
    per_prop_pred: dict[str, list[float]] = {k: [] for k in prop_keys}
    total = 0
    invalid = 0
    batch_idx = 0
    buffer_rows: list[dict] = []

    def flush_batch() -> None:
        nonlocal total, invalid, batch_idx
        if not buffer_rows:
            return

        prompts = [r["prompt"] for r in buffer_rows]
        toks = tokenize_prompts_for_generation(tokenizer, prompts, device)
        gen = model.generate(
            **toks,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_id,
        )
        gen_texts = tokenizer.batch_decode(gen, skip_special_tokens=False)
        input_ids = toks["input_ids"]
        attention_mask = toks.get("attention_mask")
        pad_token_id = tokenizer.pad_token_id

        if batch_idx == 0 and prompts:
            (report_dir / "first_batch_full_decode.txt").write_text(
                "First row of first batch (verbatim batch_decode).\n\n"
                f"prompt repr:\n{repr(prompts[0])}\n\n"
                f"generated_text repr:\n{repr(gen_texts[0])}\n",
                encoding="utf-8",
            )

        with report_csv.open("a", encoding="utf-8", newline="") as fc:
            w = csv.DictWriter(fc, fieldnames=GENERATION_REPORT_FIELDS)
            for i, (row, txt) in enumerate(zip(buffer_rows, gen_texts, strict=True)):
                total += 1
                if attention_mask is not None:
                    prompt_len = int(attention_mask[i].sum().item())
                else:
                    prompt_len = int(input_ids.shape[1])
                dbg = generation_debug_from_ids(
                    gen[i].tolist(),
                    prompt_len=prompt_len,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    max_new_tokens=max_new_tokens,
                    generated_text=txt,
                    similarity=similarity,
                )
                smi = extract_smiles(txt, similarity=similarity)
                targets = row_targets(row, task)
                preds = compute_properties_for_task(task, smi, row.get("ref_smiles"))
                joint_ok = all(preds.get(k) is not None for k in prop_keys)
                if not joint_ok:
                    invalid += 1
                else:
                    for k in prop_keys:
                        per_prop_true[k].append(float(targets[k]))
                        per_prop_pred[k].append(float(preds[k]))  # type: ignore[arg-type]

                # property_value: single float for single-prop tasks; "OK"/"NA" for multi.
                if len(prop_keys) == 1:
                    pv = "NA" if preds[prop_keys[0]] is None else preds[prop_keys[0]]
                else:
                    pv = "OK" if joint_ok else "NA"

                w.writerow(
                    {
                        "prompt": row.get("prompt"),
                        "generated_text": txt,
                        "smiles": smi,
                        "property_value": pv,
                        "targets_json": json.dumps(targets),
                        "preds_json": json.dumps(preds),
                        "qed_pred": preds.get("qed", ""),
                        "sas_pred": preds.get("sas", ""),
                        "similarity_pred": preds.get("similarity", ""),
                        **dbg,
                    }
                )

        del toks, gen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        valid = total - invalid
        per_metrics = {}
        for k in prop_keys:
            yt, yp = per_prop_true[k], per_prop_pred[k]
            per_metrics[k] = {
                "n": len(yt),
                "rmse": rmse(yt, yp) if yt else float("nan"),
                "mae": mae(yt, yp) if yt else float("nan"),
            }
        # Backward-compatible top-level mae/rmse: first property (or mean of MAEs for multi).
        if len(prop_keys) == 1:
            top_mae = per_metrics[prop_keys[0]]["mae"]
            top_rmse = per_metrics[prop_keys[0]]["rmse"]
        else:
            maes = [per_metrics[k]["mae"] for k in prop_keys if per_metrics[k]["n"]]
            rmses = [per_metrics[k]["rmse"] for k in prop_keys if per_metrics[k]["n"]]
            top_mae = float(sum(maes) / len(maes)) if maes else float("nan")
            top_rmse = float(sum(rmses) / len(rmses)) if rmses else float("nan")

        summary = {
            "model_size": model_size,
            "model_id": model_id,
            "property": task,
            "property_keys": list(prop_keys),
            "total": total,
            "valid": valid,
            "invalid": invalid,
            "valid_rate": (valid / total) if total else 0.0,
            "rmse": top_rmse,
            "mae": top_mae,
            "per_property": per_metrics,
            "updated_at_unix": int(time.time()),
        }
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            flat = {
                "model_size": model_size,
                "model_id": model_id,
                "property": task,
                "total": total,
                "valid": valid,
                "invalid": invalid,
                "valid_rate": summary["valid_rate"],
                "rmse": top_rmse,
                "mae": top_mae,
            }
            for k in prop_keys:
                flat[f"{k}_mae"] = per_metrics[k]["mae"]
                flat[f"{k}_rmse"] = per_metrics[k]["rmse"]
            w = csv.DictWriter(f, fieldnames=list(flat.keys()))
            w.writeheader()
            w.writerow(flat)
        (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        batch_idx += 1

    for row in safe_jsonl_read(prompts_path):
        buffer_rows.append(row)
        if len(buffer_rows) >= batch_size:
            flush_batch()
            buffer_rows.clear()
    flush_batch()

    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    for k in prop_keys:
        yt, yp = per_prop_true[k], per_prop_pred[k]
        if not yt:
            continue
        errs = [p - t for t, p in zip(yt, yp, strict=True)]
        metric_mae = mae(yt, yp)
        metric_rmse = rmse(yt, yp)
        title = (
            f"{model_size} / {task} / {k} "
            f"(MAE={metric_mae:.4f}, RMSE={metric_rmse:.4f}, n={len(yt)}/{total})"
        )
        plot_desired_vs_computed(
            yt,
            yp,
            report_dir / f"scatter_{k}.png",
            title=title,
            xlabel=f"desired {k}",
            ylabel=f"computed {k}",
        )
        plot_error_hist(
            errs,
            report_dir / f"error_hist_{k}.png",
            title=f"{model_size} / {task} / {k} error histogram",
        )
        # Keep legacy single scatter.png for single-property tasks.
        if len(prop_keys) == 1:
            plot_desired_vs_computed(yt, yp, report_dir / "scatter.png", title=title)
            plot_error_hist(errs, report_dir / "error_hist.png", title=f"{model_size} / {task} error histogram")

    print(json.dumps(summary))
    return summary


def write_metrics_table(summaries: list[dict], out_csv: Path) -> None:
    ensure_dir(out_csv.parent)
    tasks = []
    for s in summaries:
        t = s["property"]
        if t not in tasks:
            tasks.append(t)
    by_model: dict[str, dict[str, dict]] = {}
    for s in summaries:
        by_model.setdefault(s["model_size"], {})[s["property"]] = s

    task_keys: dict[str, list[str]] = {}
    for s in summaries:
        task_keys[s["property"]] = list(s.get("property_keys") or [s["property"]])

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["model_size"]
        for t in tasks:
            header.extend([f"{t}_mae", f"{t}_rmse", f"{t}_valid_rate"])
            keys = task_keys.get(t) or []
            if len(keys) > 1:
                for k in keys:
                    header.extend([f"{t}__{k}_mae", f"{t}__{k}_rmse"])
        w.writerow(header)
        for model, task_map in by_model.items():
            row = [model]
            for t in tasks:
                s = task_map.get(t)
                if not s:
                    # Pad empty cells for this task block.
                    keys = task_keys.get(t) or []
                    n_extra = 2 * len(keys) if len(keys) > 1 else 0
                    row.extend([""] * (3 + n_extra))
                    continue
                row.extend(
                    [
                        f"{float(s['mae']):.6f}" if s.get("mae") == s.get("mae") else "",
                        f"{float(s['rmse']):.6f}" if s.get("rmse") == s.get("rmse") else "",
                        f"{float(s['valid_rate']):.6f}",
                    ]
                )
                keys = task_keys.get(t) or []
                if len(keys) > 1:
                    per = s.get("per_property") or {}
                    for k in keys:
                        m = per.get(k) or {}
                        mae_v = m.get("mae")
                        rmse_v = m.get("rmse")
                        row.extend(
                            [
                                f"{float(mae_v):.6f}" if mae_v == mae_v else "",
                                f"{float(rmse_v):.6f}" if rmse_v == rmse_v else "",
                            ]
                        )
            w.writerow(row)
    print(f"Wrote metrics table to {out_csv}")


def main() -> None:
    quiet_rdkit()
    ap = argparse.ArgumentParser(description="Cond-Gen unified conditional generation pipeline")
    ap.add_argument("--model_size", required=True, help="e.g. chemllama-170m, chemlactica-1.3b")
    ap.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), choices=list(ALL_TASKS))
    ap.add_argument(
        "--results_base",
        type=str,
        default=None,
        help="Run root for reports/identity (prompts stay in --prompts_dir). Default: Cond-Gen/outputs/<model_size>",
    )
    ap.add_argument(
        "--prompts_dir",
        type=str,
        default=None,
        help="Shared prompt JSONL directory (default: Cond-Gen/prompts). Not copied into results.",
    )
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fewshot_k", type=int, default=0)
    ap.add_argument(
        "--mol_tag_style",
        type=str,
        default="auto",
        choices=["auto", "chemllama", "chemlactica"],
        help="auto: choose from model_size map (chemllama-* → [SMILES], chemlactica-* → [START_SMILES])",
    )
    ap.add_argument("--rebuild_prompts", action="store_true", help="Force rewrite shared prompts even if present")
    ap.add_argument("--zinc_csv", type=str, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--repetition_penalty", type=float, default=1.01)
    ap.add_argument("--do_sample", type=int, default=1)
    ap.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Re-run tasks that already have summary.json (default: skip)",
    )
    args = ap.parse_args()

    results_base = Path(
        args.results_base
        or project_path("Cond-Gen", "outputs", args.model_size)
    )
    prompts_dir = Path(args.prompts_dir) if args.prompts_dir else default_prompts_dir()
    reports_dir = results_base / "reports"
    ensure_dir(reports_dir)

    print(f"PIPELINE: model_size={args.model_size}")
    print(f"PIPELINE: results_base={results_base}")
    print(f"PIPELINE: prompts_dir={prompts_dir}")

    model, tokenizer, model_id, checkpoint_path, tokenizer_path = _load_model_and_tokenizer(
        args.model_size, args.device, args.dtype
    )
    mol_tag_style = (
        mol_tag_style_for_model(args.model_size)
        if args.mol_tag_style == "auto"
        else args.mol_tag_style
    )
    print(f"PIPELINE: mol_tag_style={mol_tag_style} (from model_size={args.model_size})")

    write_run_identity_artifacts(
        results_base / "identity",
        model=model,
        tokenizer=tokenizer,
        model_size=args.model_size,
        model_id=model_id,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        dtype=args.dtype,
        device=args.device,
        gen_settings={
            "n": args.n,
            "seed": args.seed,
            "fewshot_k": args.fewshot_k,
            "mol_tag_style": mol_tag_style,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "repetition_penalty": args.repetition_penalty,
            "do_sample": bool(args.do_sample),
            "dtype": args.dtype,
            "device": args.device,
            "tasks": list(args.tasks),
        },
        note="Cond-Gen run identity dump (compare across machines)",
    )

    zinc_csv = Path(args.zinc_csv) if args.zinc_csv else None
    manifest = ensure_prompts(
        prompts_dir=prompts_dir,
        tasks=args.tasks,
        n=args.n,
        seed=args.seed,
        fewshot_k=args.fewshot_k,
        mol_tag_style=mol_tag_style,
        zinc_csv=zinc_csv,
        rebuild=args.rebuild_prompts,
    )
    print(f"PIPELINE: using shared prompts at {prompts_dir} (manifest n={manifest['n']})")

    summaries: list[dict] = []
    for task in args.tasks:
        print(f"PIPELINE: task={task}")
        summary = run_one_task(
            model=model,
            tokenizer=tokenizer,
            model_size=args.model_size,
            model_id=model_id,
            task=task,
            prompts_path=prompts_dir / f"{task}.jsonl",
            report_dir=reports_dir / args.model_size / task,
            device=args.device,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            do_sample=bool(args.do_sample),
            seed=args.seed,
            overwrite=args.overwrite,
        )
        summaries.append(summary)

    write_metrics_table(summaries, results_base / "metrics_table.csv")
    print(f"PIPELINE: finished results_base={results_base}")


if __name__ == "__main__":
    main()
