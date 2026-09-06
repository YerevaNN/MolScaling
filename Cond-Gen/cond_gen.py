"""Conditional generation helpers: scoring, generation, scaling plots."""
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
PMO_DOCK = REPO_ROOT / "PMO-Dock"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# PMO-Dock ships the shared `benchmark` package (QED / SA / similarity).
if str(PMO_DOCK) not in sys.path:
    sys.path.insert(0, str(PMO_DOCK))

_SMILES_STRICT = re.compile(r"\[SMILES\](.*?)\[/SMILES\]", flags=re.DOTALL)
_SMILES_OPEN = re.compile(r"\[SMILES\](.*?)\[/", flags=re.DOTALL)


def load_config(path: Path | None = None) -> dict:
    cfg_path = path or (ROOT / "config.yaml")
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def task_keys(cfg: dict, task: str) -> tuple[str, ...]:
    keys = cfg["tasks"].get(task)
    if not keys:
        raise ValueError(f"Unknown task {task!r}. Known: {list(cfg['tasks'])}")
    return tuple(keys)


def resolve_model(cfg: dict, model: str) -> tuple[str, str]:
    """Return (alias_for_dirs, hub_id_or_path)."""
    models = cfg.get("models") or {}
    if model in models:
        return model, models[model]
    p = Path(model).expanduser()
    if p.exists():
        return p.name, str(p.resolve())
    return model.replace("/", "__"), model


def quiet_rdkit() -> None:
    try:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass


def mae(y_true: list[float], y_pred: list[float]) -> float:
    if not y_true:
        return float("nan")
    return float(np.mean([abs(a - b) for a, b in zip(y_true, y_pred, strict=True)]))


def rmse(y_true: list[float], y_pred: list[float]) -> float:
    if not y_true:
        return float("nan")
    return float(math.sqrt(np.mean([(a - b) ** 2 for a, b in zip(y_true, y_pred, strict=True)])))


def extract_smiles(text: str, *, similarity: bool) -> str | None:
    patterns = (_SMILES_STRICT, _SMILES_OPEN) if similarity else (_SMILES_STRICT,)
    for pat in patterns:
        m = pat.search(text)
        if m:
            smi = m.group(1).strip().replace("\n", "")
            if smi:
                return smi
    return None


def _mol(smiles: str):
    from rdkit import Chem

    try:
        m = Chem.MolFromSmiles(smiles)
        return m
    except Exception:
        return None


def _as_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _first_score(scores) -> Optional[float]:
    if scores is None or len(scores) == 0:
        return None
    return _as_float(scores[0])


def compute_qed(smiles: str) -> Optional[float]:
    from benchmark.computers.property_computers import compute_qed as bench_qed

    m = _mol(smiles)
    if m is None:
        return None
    try:
        return _first_score(bench_qed([m], verb=False))
    except Exception:
        return None


def compute_sas(smiles: str) -> Optional[float]:
    """SA score via PMO-Dock `benchmark` (Ertl/Landrum sascorer + fpscores)."""
    from benchmark.computers.property_computers import compute_sas as bench_sas

    m = _mol(smiles)
    if m is None:
        return None
    try:
        return _first_score(bench_sas([m], verb=False))
    except Exception:
        return None


def compute_similarity(ref: str, gen: str) -> Optional[float]:
    from benchmark.computers.property_computers import compute_similarity as bench_sim

    ma, mb = _mol(ref), _mol(gen)
    if ma is None or mb is None:
        return None
    try:
        return _first_score(bench_sim([mb], ma, verb=False))
    except Exception:
        return None


def score_task(keys: tuple[str, ...], smiles: str | None, ref_smiles: str | None) -> dict[str, float | None]:
    out: dict[str, float | None] = {k: None for k in keys}
    if not smiles:
        return out
    if "qed" in out:
        out["qed"] = compute_qed(smiles)
    if "sas" in out:
        out["sas"] = compute_sas(smiles)
    if "similarity" in out:
        out["similarity"] = compute_similarity(ref_smiles or "", smiles) if ref_smiles else None
    return out


def load_prompts(prompts_dir: Path, task: str) -> list[dict]:
    path = prompts_dir / f"{task}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing prompts: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _torch_dtype(name: str):
    import torch

    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get((name or "fp32").lower(), torch.float32)


def configure_tokenizer(tokenizer) -> None:
    vocab = tokenizer.get_vocab()
    pad = "<|finetune_right_pad_id|>"
    if pad in vocab:
        if getattr(tokenizer, "bos_token_id", None) is not None:
            tokenizer.add_bos_token = True
        tokenizer.pad_token_id = vocab[pad]
    elif tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"


def tokenize_prompts(tokenizer, prompts: list[str], device):
    configure_tokenizer(tokenizer)
    bos = tokenizer.bos_token_id
    ids = []
    for p in prompts:
        row = tokenizer.encode(p, add_special_tokens=False)
        if bos is not None and (not row or row[0] != bos):
            row = [bos] + row
        ids.append(row)
    return tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt").to(device)


def load_model(model_id: str, *, device: str, dtype: str, attn: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    local = os.path.isdir(model_id)
    lm = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=_torch_dtype(dtype),
        low_cpu_mem_usage=True,
        device_map={"": device},
        attn_implementation=attn,
        local_files_only=local,
        token=token,
    )
    lm.eval()
    tok = AutoTokenizer.from_pretrained(
        model_id, padding_side="left", local_files_only=local, token=token
    )
    configure_tokenizer(tok)
    return lm, tok


def eos_id(tokenizer) -> int | None:
    try:
        ids = tokenizer.encode("<|end_of_text|>", add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    except Exception:
        pass
    return tokenizer.eos_token_id


def plot_scatter(y_true, y_pred, out: Path, title: str, xlabel: str, ylabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=10, alpha=0.6)
    lo = min(min(y_true), min(y_pred))
    hi = max(max(y_true), max(y_pred))
    plt.plot([lo, hi], [lo, hi], linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def run_task(
    *,
    cfg: dict,
    model,
    tokenizer,
    model_alias: str,
    model_id: str,
    task: str,
    overwrite: bool,
) -> dict:
    keys = task_keys(cfg, task)
    io = cfg["io"]
    gen = cfg["generation"]
    prompts_dir = ROOT / io["prompts_dir"]
    report_dir = ROOT / io["results_dir"] / model_alias / task
    summary_path = report_dir / "summary.json"
    if summary_path.exists() and io.get("skip_existing", True) and not overwrite:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    rows = load_prompts(prompts_dir, task)
    similarity = "similarity" in keys
    device = gen["device"]
    batch_size = int(gen["batch_size"])
    max_new = int(gen["max_new_tokens"])
    eos = eos_id(tokenizer)

    import torch

    torch.manual_seed(int(cfg["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg["seed"]))

    per_true = {k: [] for k in keys}
    per_pred = {k: [] for k in keys}
    total = invalid = 0
    generations: list[dict] = []
    report_dir.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        toks = tokenize_prompts(tokenizer, [r["prompt"] for r in batch], device)
        out_ids = model.generate(
            **toks,
            max_new_tokens=max_new,
            do_sample=bool(gen["do_sample"]),
            temperature=float(gen["temperature"]),
            repetition_penalty=float(gen["repetition_penalty"]),
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos,
        )
        texts = tokenizer.batch_decode(out_ids, skip_special_tokens=False)
        for row, txt in zip(batch, texts, strict=True):
            total += 1
            smi = extract_smiles(txt, similarity=similarity)
            targets = {str(k): float(v) for k, v in (row.get("targets") or {}).items()}
            preds = score_task(keys, smi, row.get("ref_smiles"))
            ok = all(preds[k] is not None for k in keys)
            if not ok:
                invalid += 1
            else:
                for k in keys:
                    per_true[k].append(float(targets[k]))
                    per_pred[k].append(float(preds[k]))  # type: ignore[arg-type]
            if io.get("save_generations", True):
                generations.append(
                    {
                        "id": row.get("id"),
                        "prompt": row["prompt"],
                        "generated_text": txt,
                        "smiles": smi,
                        "targets": targets,
                        "preds": preds,
                    }
                )
        del toks, out_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    per_metrics = {
        k: {
            "n": len(per_true[k]),
            "mae": mae(per_true[k], per_pred[k]),
            "rmse": rmse(per_true[k], per_pred[k]),
        }
        for k in keys
    }
    maes = [per_metrics[k]["mae"] for k in keys if per_metrics[k]["n"]]
    rmses = [per_metrics[k]["rmse"] for k in keys if per_metrics[k]["n"]]
    summary = {
        "model_size": model_alias,
        "model_id": model_id,
        "property": task,
        "property_keys": list(keys),
        "total": total,
        "valid": total - invalid,
        "invalid": invalid,
        "valid_rate": (total - invalid) / total if total else 0.0,
        "mae": float(sum(maes) / len(maes)) if maes else float("nan"),
        "rmse": float(sum(rmses) / len(rmses)) if rmses else float("nan"),
        "per_property": per_metrics,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if generations:
        with (report_dir / "generations.jsonl").open("w", encoding="utf-8") as f:
            for g in generations:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
    if io.get("save_scatter", True):
        for k in keys:
            if per_true[k]:
                plot_scatter(
                    per_true[k],
                    per_pred[k],
                    report_dir / f"scatter_{k}.png",
                    title=f"{model_alias} / {task} / {k}",
                    xlabel=f"desired {k}",
                    ylabel=f"computed {k}",
                )
    return summary


def _load_summaries(results_dir: Path) -> dict[tuple[str, str], dict]:
    out = {}
    for path in results_dir.rglob("summary.json"):
        try:
            s = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if s.get("model_size") and s.get("property"):
            out[(s["model_size"], s["property"])] = s
    return out


def _mae_of(summary: dict, prop: str) -> float | None:
    per = summary.get("per_property") or {}
    if prop in per and per[prop].get("mae") is not None:
        v = float(per[prop]["mae"])
        return v if v == v else None
    return None


def write_scaling_plots(cfg: dict, results_dir: Path, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_cfg = cfg["plot"]
    order = list(plot_cfg["model_order"])
    labels = plot_cfg["model_labels"]
    sas_norm = float(plot_cfg["sas_norm"])
    summaries = _load_summaries(results_dir)
    if not summaries:
        raise SystemExit(f"No summary.json under {results_dir}")

    styles = {
        "qed": {"color": "#1f77b4", "marker": "o", "ylabel": "MAE"},
        "sas": {"color": "#ff7f0e", "marker": "s", "ylabel": "MAE"},
        "similarity": {"color": "#2ca02c", "marker": "^", "ylabel": "MAE"},
        "norm": {"color": "#9467bd", "marker": "D", "ylabel": "QED MAE + SA MAE / 5"},
    }
    arity_task = {
        "single": {"qed": "qed", "sas": "sas", "similarity": "similarity_random"},
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

    def series(arity: str, prop: str) -> list[float | None]:
        task = arity_task[arity][prop]
        vals = []
        for m in order:
            s = summaries.get((m, task))
            vals.append(_mae_of(s, prop) if s else None)
        return vals

    def norm_series(arity: str) -> list[float | None]:
        out = []
        for q, s in zip(series(arity, "qed"), series(arity, "sas"), strict=True):
            out.append(None if q is None or s is None else q + s / sas_norm)
        return out

    def draw(ax, ys, kind: str, title: str) -> None:
        st = styles[kind]
        xs = list(range(len(order)))
        x_ok = [x for x, y in zip(xs, ys, strict=True) if y is not None]
        y_ok = [y for y in ys if y is not None]
        if y_ok:
            ax.plot(x_ok, y_ok, color=st["color"], marker=st["marker"], markersize=8, linewidth=1.8)
            span = max(y_ok) - min(y_ok)
            pad = 0.04 * span if span > 0 else 0.004 * (abs(max(y_ok)) + 1e-6)
            for x, y in zip(xs, ys, strict=True):
                if y is not None:
                    ax.text(x, y + pad, f"{y:.3f}", ha="center", va="bottom", fontsize=9)
            p = 0.22 * span if span > 0 else 0.08 * (abs(max(y_ok)) + 1e-6)
            ax.set_ylim(min(y_ok) - 0.4 * p, max(y_ok) + p)
        ax.set_title(title)
        ax.set_xlabel("Model size")
        ax.set_ylabel(st["ylabel"])
        ax.set_xticks(range(len(order)), [labels.get(m, m) for m in order])
        ax.grid(True, linestyle="--", linewidth=0.8, color="0.75")
        ax.set_axisbelow(True)

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2))
    draw(axes[0], series("single", "qed"), "qed", "QED scaling")
    draw(axes[1], series("single", "sas"), "sas", "SA scaling")
    draw(axes[2], series("single", "similarity"), "similarity", "Similarity scaling")
    fig.tight_layout()
    fig.savefig(out_dir / "scaling_single.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.4))
    for r, (arity, order_name) in enumerate((("double", "QED→SA"), ("double_swap", "SA→QED"))):
        draw(axes[r][0], series(arity, "qed"), "qed", f"QED ({order_name})")
        draw(axes[r][1], series(arity, "sas"), "sas", f"SA ({order_name})")
        draw(axes[r][2], norm_series(arity), "norm", f"Norm. avg (QED + SA/5) ({order_name})")
    fig.tight_layout()
    fig.savefig(out_dir / "scaling_double.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.4))
    for r, (arity, order_name) in enumerate((("triple", "QED→SA"), ("triple_swap", "SA→QED"))):
        draw(axes[r][0], series(arity, "qed"), "qed", f"QED ({order_name}, triple)")
        draw(axes[r][1], series(arity, "sas"), "sas", f"SA ({order_name}, triple)")
        draw(axes[r][2], series(arity, "similarity"), "similarity", f"Similarity ({order_name}, triple)")
    fig.tight_layout()
    fig.savefig(out_dir / "scaling_triple.png", dpi=200)
    plt.close(fig)
    print(f"Wrote plots to {out_dir}")
