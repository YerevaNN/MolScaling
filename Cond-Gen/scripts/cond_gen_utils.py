from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import torch

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, QED


@dataclass(frozen=True)
class ZincRow:
    smiles: str
    qed: float
    sas: float


def repo_root() -> Path:
    # Prefer PROJECT_ROOT when running on Slurm.
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root)
    return Path(__file__).resolve().parents[2]


def project_path(*parts: str) -> Path:
    return repo_root().joinpath(*parts)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def quiet_rdkit() -> None:
    """
    Best-effort suppression of RDKit stderr noise during bulk scoring.
    Safe to call even if RDKit logging APIs are unavailable.
    """
    try:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass


def is_galactica_style_tokenizer(tokenizer) -> bool:
    """
    ChemLactica / Galactica / Chemma-style tokenizers (OPT or Gemma heritage).

    These use <pad>/<s>/</s> (or Gemma equivalents) rather than Llama-3 chem
    specials like <|begin_of_text|> / <|finetune_right_pad_id|>.
    """
    vocab = tokenizer.get_vocab()
    if "<|finetune_right_pad_id|>" in vocab or "<|begin_of_text|>" in vocab:
        return False
    if "<pad>" in vocab and "</s>" in vocab:
        return True
    # Galactica / ChemLactica also ship atomic [START_SMILES] specials.
    try:
        ids = tokenizer.encode("[START_SMILES]", add_special_tokens=False)
        if len(ids) == 1:
            return True
    except Exception:
        pass
    return False


def document_start_token_id(tokenizer) -> int | None:
    """
    Leading special id prepended to every Cond-Gen prompt.

    - Llama-3 chem: BOS (<|begin_of_text|>)
    - ChemLactica / Galactica: EOS (</s>), matching genetic_chemalactica
    """
    if is_galactica_style_tokenizer(tokenizer):
        return getattr(tokenizer, "eos_token_id", None)
    return getattr(tokenizer, "bos_token_id", None)


def configure_tokenizer_for_generation(tokenizer) -> None:
    """
    Normalize pad / BOS settings for Cond-Gen generation.

    Llama-3 chem: force BOS and <|finetune_right_pad_id|> when present.
    ChemLactica / Galactica: keep <pad> (already set) and do not invent Llama pads.
    """
    vocab = tokenizer.get_vocab()
    llama_pad = "<|finetune_right_pad_id|>"
    if llama_pad in vocab:
        if getattr(tokenizer, "bos_token_id", None) is not None:
            tokenizer.add_bos_token = True
        tokenizer.pad_token_id = vocab[llama_pad]
    elif "<pad>" in vocab:
        tokenizer.pad_token_id = vocab["<pad>"]
    elif tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


def tokenize_prompts_for_generation(tokenizer, prompts: list[str], device):
    """
    Batch-encode prompts with a leading document-start special (training parity).
    Uses left padding when the tokenizer is configured for it.
    """
    start_id = document_start_token_id(tokenizer)
    input_ids: list[list[int]] = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if start_id is not None and (not ids or ids[0] != start_id):
            ids = [start_id] + ids
        input_ids.append(ids)

    batch = tokenizer.pad(
        {"input_ids": input_ids},
        padding=True,
        return_tensors="pt",
    )
    return batch.to(device)


def write_tokenization_probe(path: Path, tokenizer, *, note: str = "") -> None:
    """
    Log how the tokenizer encodes a Cond-Gen-style prompt (BOS / specials visible in decode).
    Intended for result dirs so runs document whether the document-start token is inserted.
    """
    configure_tokenizer_for_generation(tokenizer)
    sample = "[QED]0.78[/QED][SMILES]"
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    start_id = document_start_token_id(tokenizer)
    gen_ids = tokenize_prompts_for_generation(tokenizer, [sample], device="cpu")[
        "input_ids"
    ][0].tolist()
    dec_gen = tokenizer.decode(gen_ids, skip_special_tokens=False)
    leading_start = bool(gen_ids and start_id is not None and gen_ids[0] == start_id)
    lines = [
        note.strip(),
        f"tokenizer.name_or_path: {getattr(tokenizer, 'name_or_path', '')}",
        f"galactica_style={is_galactica_style_tokenizer(tokenizer)}",
        f"add_bos_token={getattr(tokenizer, 'add_bos_token', 'N/A')}",
        f"bos_token_id={bos_id} eos_token_id={eos_id} document_start_token_id={start_id}",
        f"pad_token_id={getattr(tokenizer, 'pad_token_id', None)}",
        "",
        "Sample prompt (repr):",
        repr(sample),
        "",
        "tokenize_prompts_for_generation (used by generation runners):",
        f"  len={len(gen_ids)} ids[:32]={gen_ids[:32]}",
        f"  decode(skip_special_tokens=False) prefix (first 200 chars repr):",
        repr(dec_gen[:200]),
        f"  leading_token_is_document_start: {leading_start}",
        "",
        "batch_decode in runners uses skip_special_tokens=False so any BOS/EOS present in ids appears verbatim.",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _tensor_fingerprint(t: torch.Tensor) -> dict:
    x = t.detach().float().cpu().contiguous().view(-1)
    n = int(x.numel())
    # Stable content hash of full tensor bytes (float32).
    digest = hashlib.sha256(x.numpy().tobytes()).hexdigest()
    head = x[:8].tolist() if n else []
    tail = x[-8:].tolist() if n else []
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "numel": n,
        "sha256_f32": digest,
        "mean": float(x.mean().item()) if n else None,
        "std": float(x.std(unbiased=False).item()) if n else None,
        "min": float(x.min().item()) if n else None,
        "max": float(x.max().item()) if n else None,
        "head8": head,
        "tail8": tail,
    }


def _named_param_fingerprints(model, names: list[str] | None = None) -> dict:
    """Fingerprint a few named parameters for cross-machine weight identity checks."""
    state = dict(model.named_parameters())
    if names is None:
        # Prefer a small, informative set if present.
        candidates = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "lm_head.weight",
        ]
        names = [n for n in candidates if n in state]
        if not names:
            # Fallback: first / middle / last parameter names.
            keys = list(state.keys())
            if keys:
                mid = keys[len(keys) // 2]
                names = list(dict.fromkeys([keys[0], mid, keys[-1]]))
    out: dict = {}
    for name in names:
        if name not in state:
            out[name] = {"missing": True}
            continue
        out[name] = _tensor_fingerprint(state[name].data)
    return out


def _checkpoint_file_hashes(checkpoint_path: str) -> dict:
    p = Path(checkpoint_path)
    if not p.is_dir():
        return {"kind": "hub_or_missing", "path": checkpoint_path, "files": {}}
    files = {}
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
    ):
        fp = p / name
        if fp.exists():
            files[name] = {
                "size_bytes": fp.stat().st_size,
                "sha256": _sha256_file(fp),
            }
    # Also hash any shard safetensors if present.
    for shard in sorted(p.glob("model-*.safetensors")):
        files[shard.name] = {
            "size_bytes": shard.stat().st_size,
            "sha256": _sha256_file(shard),
        }
    return {"kind": "local_dir", "path": str(p.resolve()), "files": files}


@torch.inference_mode()
def _fixed_prompt_generation_probe(
    model,
    tokenizer,
    *,
    prompt: str = "[QED]0.78[/QED][SMILES]",
    max_new_tokens: int = 64,
    seed: int = 1,
) -> dict:
    """
    Deterministic (greedy) generation for a fixed prompt.
    Compare `generated_ids` / `generated_text` across machines.
    """
    configure_tokenizer_for_generation(tokenizer)
    device = str(next(model.parameters()).device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    toks = tokenize_prompts_for_generation(tokenizer, [prompt], device)
    eos_ids = None
    try:
        eot = tokenizer.encode("<|end_of_text|>", add_special_tokens=False)
        if len(eot) == 1:
            eos_ids = int(eot[0])
    except Exception:
        pass
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id

    gen = model.generate(
        **toks,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        repetition_penalty=1.0,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_ids,
    )
    gen_ids = gen[0].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    prompt_len = int(toks["attention_mask"][0].sum().item()) if "attention_mask" in toks else int(
        toks["input_ids"].shape[1]
    )
    new_ids = gen_ids[prompt_len:]
    return {
        "prompt": prompt,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": 1.0,
        "repetition_penalty": 1.0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": eos_ids,
        "prompt_ids": toks["input_ids"][0].tolist(),
        "prompt_len": prompt_len,
        "generated_ids": gen_ids,
        "new_token_ids": new_ids,
        "generated_text": text,
        "extracted_smiles": extract_smiles(text, similarity=False),
    }


def write_run_identity_artifacts(
    out_dir: Path,
    *,
    model,
    tokenizer,
    model_size: str,
    model_id: str,
    checkpoint_path: str,
    tokenizer_path: str,
    dtype: str,
    device: str,
    gen_settings: dict | None = None,
    note: str = "",
) -> dict:
    """
    Write machine/model identity artifacts for cross-machine collapse debugging.

    Files:
      - model_identity.json  (machine-readable)
      - model_identity.txt   (human-readable summary)
      - fixed_prompt_greedy.json  (deterministic generation probe)
    """
    ensure_dir(out_dir)
    configure_tokenizer_for_generation(tokenizer)

    try:
        import transformers
        transformers_version = transformers.__version__
    except Exception:
        transformers_version = None
    try:
        import rdkit
        rdkit_version = getattr(rdkit, "__version__", None)
    except Exception:
        rdkit_version = None

    sample_prompt = "[QED]0.78[/QED][SMILES]"
    sample_ids = tokenize_prompts_for_generation(tokenizer, [sample_prompt], device="cpu")[
        "input_ids"
    ][0].tolist()

    cfg = getattr(model, "config", None)
    config_subset = {}
    if cfg is not None:
        for k in (
            "model_type",
            "architectures",
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
            "rms_norm_eps",
            "rope_theta",
            "tie_word_embeddings",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "torch_dtype",
        ):
            if hasattr(cfg, k):
                v = getattr(cfg, k)
                config_subset[k] = v if not isinstance(v, torch.dtype) else str(v)

    identity = {
        "note": note,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers_version,
        "rdkit": rdkit_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "env": {
            "PROJECT_ROOT": os.environ.get("PROJECT_ROOT"),
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
            "HF_LOCAL_MODEL_DIR": os.environ.get("HF_LOCAL_MODEL_DIR"),
            "HF_LOCAL_TOKENIZER_DIR": os.environ.get("HF_LOCAL_TOKENIZER_DIR"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "model_size": model_size,
        "model_id": model_id,
        "checkpoint_path": checkpoint_path,
        "tokenizer_path": tokenizer_path,
        "dtype_arg": dtype,
        "device_arg": device,
        "param_dtype": str(next(model.parameters()).dtype),
        "param_device": str(next(model.parameters()).device),
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
        "model_config": config_subset,
        "checkpoint_files": _checkpoint_file_hashes(checkpoint_path),
        "tokenizer_files": _checkpoint_file_hashes(tokenizer_path),
        "tokenizer": {
            "name_or_path": getattr(tokenizer, "name_or_path", None),
            "vocab_size": getattr(tokenizer, "vocab_size", None),
            "bos_token": getattr(tokenizer, "bos_token", None),
            "eos_token": getattr(tokenizer, "eos_token", None),
            "pad_token": getattr(tokenizer, "pad_token", None),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "add_bos_token": getattr(tokenizer, "add_bos_token", None),
            "padding_side": getattr(tokenizer, "padding_side", None),
            "document_start_token_id": document_start_token_id(tokenizer),
            "galactica_style": is_galactica_style_tokenizer(tokenizer),
            "special_tokens_map": getattr(tokenizer, "special_tokens_map", None),
        },
        "sample_tokenization": {
            "prompt": sample_prompt,
            "ids": sample_ids,
            "decode": tokenizer.decode(sample_ids, skip_special_tokens=False),
        },
        "weight_fingerprints": _named_param_fingerprints(model),
        "generation_settings": gen_settings or {},
    }

    probe = _fixed_prompt_generation_probe(model, tokenizer)
    identity["fixed_prompt_greedy"] = {
        "prompt": probe["prompt"],
        "new_token_ids": probe["new_token_ids"],
        "extracted_smiles": probe["extracted_smiles"],
        "generated_text_prefix": probe["generated_text"][:500],
    }

    json_path = out_dir / "model_identity.json"
    probe_path = out_dir / "fixed_prompt_greedy.json"
    txt_path = out_dir / "model_identity.txt"

    json_path.write_text(json.dumps(identity, indent=2, default=str) + "\n", encoding="utf-8")
    probe_path.write_text(json.dumps(probe, indent=2, default=str) + "\n", encoding="utf-8")

    lines = [
        f"hostname: {identity['hostname']}",
        f"torch={identity['torch']}  transformers={identity['transformers']}  rdkit={identity['rdkit']}",
        f"cuda: {identity['cuda_device_name']} (count={identity['cuda_device_count']})",
        f"model_size={model_size}  model_id={model_id}",
        f"checkpoint_path={checkpoint_path}",
        f"tokenizer_path={tokenizer_path}",
        f"HF_LOCAL_MODEL_DIR={identity['env']['HF_LOCAL_MODEL_DIR']}",
        f"param_dtype={identity['param_dtype']}  device={identity['param_device']}  n_params={identity['num_parameters']}",
        "",
        "tokenizer ids:",
        f"  bos={identity['tokenizer']['bos_token_id']} eos={identity['tokenizer']['eos_token_id']} "
        f"pad={identity['tokenizer']['pad_token_id']} doc_start={identity['tokenizer']['document_start_token_id']}",
        "",
        f"sample prompt ids: {sample_ids}",
        f"sample decode: {identity['sample_tokenization']['decode']!r}",
        "",
        "checkpoint file sha256:",
    ]
    for name, meta in (identity["checkpoint_files"].get("files") or {}).items():
        lines.append(f"  {name}: {meta.get('sha256')}  ({meta.get('size_bytes')} bytes)")
    lines.append("")
    lines.append("weight fingerprints (sha256_f32):")
    for name, meta in identity["weight_fingerprints"].items():
        if meta.get("missing"):
            lines.append(f"  {name}: MISSING")
        else:
            lines.append(f"  {name}: {meta['sha256_f32']}  shape={meta['shape']}")
    lines.append("")
    lines.append("fixed greedy probe:")
    lines.append(f"  new_token_ids={probe['new_token_ids'][:64]}{'...' if len(probe['new_token_ids'])>64 else ''}")
    lines.append(f"  extracted_smiles={probe['extracted_smiles']!r}")
    lines.append(f"  text_prefix={probe['generated_text'][:240]!r}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"PIPELINE: wrote identity artifacts to {out_dir}")
    return identity


def read_zinc250k_rows(csv_path: Path) -> list[ZincRow]:
    """
    Robust loader for `genmol/data/zinc250k.csv` which contains quoted SMILES that
    can span lines (embedded newlines).
    """
    rows: list[ZincRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            smi = (r.get("smiles") or "").replace("\n", "").strip()
            if not smi:
                continue
            try:
                qed = float(r["qed"])
                sas = float(r["SAS"])
            except Exception:
                continue
            rows.append(ZincRow(smiles=smi, qed=qed, sas=sas))
    return rows


def rdkit_mol(smiles: str) -> Optional[Chem.Mol]:
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        return m
    except Exception:
        return None


def compute_qed(smiles: str) -> Optional[float]:
    m = rdkit_mol(smiles)
    if m is None:
        return None
    try:
        return float(QED.qed(m))
    except Exception:
        return None


def _find_fpscores_path() -> Optional[Path]:
    """
    SA score implementation in this repo expects an `fpscores.pkl.gz` file.
    We try a few common locations.
    """
    # 1) If we vendor it under Cond-Gen/assets/
    p = project_path("Cond-Gen", "assets", "fpscores.pkl.gz")
    if p.exists():
        return p

    # 2) If it exists next to the repo's sascorer.py (older layouts)
    p = project_path("genetic_chemalactica", "oracles", "synthesizability", "fpscores.pkl.gz")
    if p.exists():
        return p

    # 3) If it exists next to benchmark implementation
    p = project_path("benchmark", "synthesizability", "fpscores.pkl.gz")
    if p.exists():
        return p

    # 4) Try RDKit contrib path if installed (best effort)
    try:
        import rdkit

        for base in rdkit.__path__:
            cand = Path(base) / "Contrib" / "SA_Score" / "fpscores.pkl.gz"
            if cand.exists():
                return cand
    except Exception:
        return None

    return None


def _maybe_download_fpscores(to_path: Path) -> bool:
    """
    Best-effort download of fpscores from RDKit's repository if it's missing.
    This is only used as a fallback in cluster environments where the RDKit
    Contrib data may not be installed.
    """
    try:
        import urllib.request

        ensure_dir(to_path.parent)
        # RDKit Contrib SA_Score data (binary). If network is blocked, this will fail gracefully.
        url = "https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/fpscores.pkl.gz"
        urllib.request.urlretrieve(url, to_path)  # noqa: S310 (controlled URL)
        return to_path.exists() and to_path.stat().st_size > 0
    except Exception:
        return False


def _torch_dtype_from_str(dtype: str):
    dt = (dtype or "").lower()
    if dt in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dt in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def load_hf_model_and_tokenizer(
    *,
    checkpoint_path: str,
    tokenizer_path: str,
    device: str,
    dtype: str,
    padding_side: str = "left",
):
    """
    Shared loader for Cond-Gen scripts.

    Handles:
    - dtype/device_map setup
    - local_files_only inference for local checkpoint/tokenizer dirs
    - HF token env vars
    - fallback tokenizer construction for some exported checkpoints
    - Cond-Gen tokenizer config normalization (BOS/pad)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    local_files_only = os.path.isdir(checkpoint_path) and os.path.isdir(tokenizer_path)

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=_torch_dtype_from_str(dtype),
        low_cpu_mem_usage=True,
        device_map={"": device},
        attn_implementation="sdpa",
        local_files_only=local_files_only,
        token=hf_token,
    )
    model.eval()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            padding_side=padding_side,
            local_files_only=local_files_only,
            token=hf_token,
        )
    except ValueError as e:
        # Some exported checkpoints have `tokenizer_config.json` pointing at a backend
        # class that isn't registered in the current Transformers build.
        if "TokenizersBackend" not in str(e):
            raise

        from huggingface_hub import hf_hub_download

        if os.path.isdir(tokenizer_path):
            cfg_path = os.path.join(tokenizer_path, "tokenizer_config.json")
            tok_file = os.path.join(tokenizer_path, "tokenizer.json")
        else:
            cfg_path = hf_hub_download(tokenizer_path, "tokenizer_config.json", token=hf_token)
            tok_file = hf_hub_download(tokenizer_path, "tokenizer.json", token=hf_token)

        with open(cfg_path, "r", encoding="utf-8") as f:
            tok_cfg = json.load(f)

        bos_token = tok_cfg.get("bos_token")
        eos_token = tok_cfg.get("eos_token")
        model_max_length = tok_cfg.get("model_max_length")

        from transformers import PreTrainedTokenizerFast

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tok_file,
            bos_token=bos_token,
            eos_token=eos_token,
            padding_side=padding_side,
            model_max_length=model_max_length,
        )

    configure_tokenizer_for_generation(tokenizer)
    return model, tokenizer


def _load_sascorer():
    # Use the benchmark implementation (requested).
    # Some scripts are executed from `Cond-Gen/scripts/`, in which case the repo root
    # may not be on `sys.path` and `import benchmark...` would fail.
    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)

    from benchmark.synthesizability import sascorer as repo_sascorer

    fpscores = _find_fpscores_path()
    if fpscores is None:
        # Try to download into Cond-Gen/assets/ if we can.
        vendored = project_path("Cond-Gen", "assets", "fpscores.pkl.gz")
        if _maybe_download_fpscores(vendored):
            fpscores = vendored
    if fpscores is None:
        raise FileNotFoundError(
            "Could not find fpscores.pkl.gz needed for SAS computation. "
            "Expected one of: "
            "`Cond-Gen/assets/fpscores.pkl.gz` or RDKit's `Contrib/SA_Score/fpscores.pkl.gz`."
        )

    # repo_sascorer expects a basename without the `.pkl.gz` suffix.
    name_no_suffix = str(fpscores).removesuffix(".pkl.gz")
    repo_sascorer.readFragmentScores(name_no_suffix)
    return repo_sascorer


_SASCORER = None


def compute_sas(smiles: str) -> Optional[float]:
    global _SASCORER
    m = rdkit_mol(smiles)
    if m is None:
        return None
    try:
        if _SASCORER is None:
            _SASCORER = _load_sascorer()
        return float(_SASCORER.calculateScore(m))
    except Exception:
        return None


def _morgan_fp(m: Chem.Mol, radius: int = 2, n_bits: int = 2048):
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)


def compute_similarity(smiles_a: str, smiles_b: str) -> Optional[float]:
    ma = rdkit_mol(smiles_a)
    mb = rdkit_mol(smiles_b)
    if ma is None or mb is None:
        return None
    try:
        fa = _morgan_fp(ma)
        fb = _morgan_fp(mb)
        return float(DataStructs.TanimotoSimilarity(fa, fb))
    except Exception:
        return None


_QED_RE = re.compile(r"\[QED\]([0-9]*\.?[0-9]+)\[/QED\]")
_SAS_RE = re.compile(r"\[SAS\]([0-9]*\.?[0-9]+)\[/SAS\]")
_SIM_RE = re.compile(r"\[SIMILAR\](.*?)\[/SIMILAR\]")


def parse_target_from_prompt(prompt: str, task: str) -> float | None:
    """
    Extract the numeric target value embedded in the prompt.

    - qed: [QED]{val}[/QED]
    - sas: [SAS]{val}[/SAS]
    - similarity_random: [SIMILAR]{ref_smiles} {val}[/SIMILAR]
    """
    if task == "qed":
        m = _QED_RE.search(prompt)
        return float(m.group(1)) if m else None
    if task == "sas":
        m = _SAS_RE.search(prompt)
        return float(m.group(1)) if m else None
    if task == "similarity_random":
        m = _SIM_RE.search(prompt)
        if not m:
            return None
        content = m.group(1).strip()
        parts = content.rsplit(" ", 1)
        if len(parts) != 2:
            return None
        try:
            return float(parts[1])
        except Exception:
            return None
    raise ValueError(task)


def parse_ref_smiles_from_prompt(prompt: str, task: str) -> str | None:
    """
    Extract the reference SMILES from similarity_random prompts.
    Content format: "{ref_smiles} {target}"
    """
    if task != "similarity_random":
        return None
    m = _SIM_RE.search(prompt)
    if not m:
        return None
    content = m.group(1).strip()
    parts = content.rsplit(" ", 1)
    if len(parts) != 2:
        return None
    return parts[0].strip() or None


def safe_jsonl_write(path: Path, rows: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def safe_jsonl_read(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# Prefer ChemLactica / Galactica [START_SMILES]...[END_SMILES], then ChemLlama [SMILES].
_SMILES_RE_START_END = re.compile(r"\[START_SMILES\](.*?)\[END_SMILES\]", flags=re.DOTALL)
_SMILES_RE_START_OPEN = re.compile(r"\[START_SMILES\](.*?)\[/", flags=re.DOTALL)
_SMILES_RE_STRICT = re.compile(r"\[SMILES\](.*?)\[/SMILES\]", flags=re.DOTALL)
# Similarity generations often close with `[/SIMILAR]` (or another tag) instead of a mol closer.
_SMILES_RE_SIMILARITY = re.compile(r"\[SMILES\](.*?)\[/", flags=re.DOTALL)


# Canonical columns for `generation_report.csv` (streaming runs + fix script).
# Downstream tools may rely on this order; add new debug columns at the end only.
GENERATION_REPORT_FIELDS = [
    "prompt",
    "generated_text",
    "smiles",
    "property_value",
    "targets_json",
    "preds_json",
    "qed_pred",
    "sas_pred",
    "similarity_pred",
    "ended_with_eos",
    "hit_max_new_tokens",
    "contains_closing_tag",
    "new_tokens",
]
GENERATION_REPORT_FIELDS_SET = frozenset(GENERATION_REPORT_FIELDS)


def contains_closing_tag(generated_text: str, *, similarity: bool = False) -> bool:
    tags = (
        ("[/SMILES]", "[END_SMILES]", "[/SIMILAR]", "[/SIMILARITY]")
        if similarity
        else ("[/SMILES]", "[END_SMILES]")
    )
    return any(tag in generated_text for tag in tags)


def generation_debug_from_ids(
    generated_ids: list[int],
    *,
    prompt_len: int,
    pad_token_id: int | None,
    eos_token_id: int | None,
    max_new_tokens: int,
    generated_text: str = "",
    similarity: bool = False,
) -> dict[str, int]:
    """
    Infer how a single sequence ended from token ids (post-generate).
    """
    cont = generated_ids[int(prompt_len) :]
    if pad_token_id is not None:
        while cont and cont[-1] == pad_token_id:
            cont = cont[:-1]

    new_tokens = len(cont)
    hit_max = int(new_tokens >= max_new_tokens)
    ended_eos = 0
    if eos_token_id is not None and cont and cont[-1] == eos_token_id:
        ended_eos = 1
    elif "<|end_of_text|>" in generated_text:
        ended_eos = 1

    return {
        "ended_with_eos": ended_eos,
        "hit_max_new_tokens": hit_max,
        "contains_closing_tag": int(contains_closing_tag(generated_text, similarity=similarity)),
        "new_tokens": new_tokens,
    }


def extract_smiles(generated_text: str, *, similarity: bool = False) -> str | None:
    """
    Extract the first generated SMILES span.

    Prefers ChemLactica / Galactica `[START_SMILES]...[END_SMILES]`, then ChemLlama
    `[SMILES]...[/SMILES]`. Similarity runs often emit `[/SIMILAR]` (or another
    closing tag), so open-ended patterns stop at `[/`.
    """
    if similarity:
        patterns = (
            _SMILES_RE_START_END,
            _SMILES_RE_START_OPEN,
            _SMILES_RE_STRICT,
            _SMILES_RE_SIMILARITY,
        )
    else:
        patterns = (_SMILES_RE_START_END, _SMILES_RE_STRICT)
    for pattern in patterns:
        m = pattern.search(generated_text)
        if m:
            smi = m.group(1).strip().replace("\n", "")
            if smi:
                return smi
    return None


def rmse(y_true: list[float], y_pred: list[float]) -> float:
    assert len(y_true) == len(y_pred)
    if len(y_true) == 0:
        return float("nan")
    mse = float(np.mean([(a - b) ** 2 for a, b in zip(y_true, y_pred, strict=True)]))
    return math.sqrt(mse)


def mae(y_true: list[float], y_pred: list[float]) -> float:
    assert len(y_true) == len(y_pred)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean([abs(a - b) for a, b in zip(y_true, y_pred, strict=True)]))


# ChemLlama uses [SMILES]/[/SMILES]; ChemLactica / Galactica / Chemma use
# [START_SMILES]/[END_SMILES] (atomic specials in those tokenizers).
MOL_TAG_STYLES = {
    "chemllama": ("[SMILES]", "[/SMILES]"),
    "chemlactica": ("[START_SMILES]", "[END_SMILES]"),
}

# Model key → mol delimiter style used when building Cond-Gen prompts.
MODEL_MOL_TAG_STYLE = {
    "chemllama-170m": "chemllama",
    "chemllama-380m": "chemllama",
    "chemllama-1b": "chemllama",
    "chemllama-3b": "chemllama",
    "chemlactica-1.3b": "chemlactica",
}

# Predefined experiments:
#   single: qed | sas | similarity
#   double: qed+sa (canonical tag order) and sa+qed (same ZINC pair, tags swapped)
#   triple: qed+sa+sim and sa+qed+sim (same values; only QED/SAS tag order changes)
# QED and SAS in any joint prompt always come from one real ZINC molecule.
TASK_PROPERTY_KEYS: dict[str, tuple[str, ...]] = {
    "qed": ("qed",),
    "sas": ("sas",),
    "similarity_random": ("similarity",),
    "qed_sas": ("qed", "sas"),
    "sas_qed": ("sas", "qed"),
    "qed_sas_similarity_random": ("qed", "sas", "similarity"),
    "sas_qed_similarity_random": ("sas", "qed", "similarity"),
    # Extra pair tasks from earlier sweeps (not in the predefined grid).
    "qed_similarity_random": ("qed", "similarity"),
    "sas_similarity_random": ("sas", "similarity"),
}
SINGLE_TASKS = ("qed", "sas", "similarity_random")
DOUBLE_TASKS = ("qed_sas", "sas_qed")
TRIPLE_TASKS = ("qed_sas_similarity_random", "sas_qed_similarity_random")
DEFAULT_TASKS = SINGLE_TASKS
PREDEFINED_TASKS = SINGLE_TASKS + DOUBLE_TASKS + TRIPLE_TASKS
MULTI_TASKS = DOUBLE_TASKS + TRIPLE_TASKS + ("qed_similarity_random", "sas_similarity_random")
ALL_TASKS = PREDEFINED_TASKS + ("qed_similarity_random", "sas_similarity_random")

# Distinct RNG offsets. Order-swap tasks reuse the canonical offset so they
# condition on the same ZINC (QED, SAS) pair (and the same random similarity).
TASK_SEED_OFFSET = {
    "qed": 0,
    "sas": 1,
    "similarity_random": 3,
    "qed_sas": 4,
    "sas_qed": 4,
    "qed_sas_similarity_random": 5,
    "sas_qed_similarity_random": 5,
    "qed_similarity_random": 6,
    "sas_similarity_random": 7,
}


def default_prompts_dir() -> Path:
    """Shared in-repo prompt files (not copied into result directories)."""
    return project_path("Cond-Gen", "prompts")


def task_property_keys(task: str) -> tuple[str, ...]:
    """Ordered property keys evaluated for a task."""
    keys = TASK_PROPERTY_KEYS.get(task)
    if keys is None:
        raise ValueError(f"Unsupported task: {task}")
    return keys


def is_multi_property_task(task: str) -> bool:
    return task in MULTI_TASKS


def task_uses_similarity_extraction(task: str) -> bool:
    return "similarity" in task_property_keys(task)



def mol_tag_style_for_model(model_size: str) -> str:
    """
    Pick mol open/close tags from the model family.

    ChemLlama → [SMILES] / [/SMILES]
    ChemLactica (and other Galactica-style keys) → [START_SMILES] / [END_SMILES]
    """
    key = (model_size or "").strip().lower()
    if key in MODEL_MOL_TAG_STYLE:
        return MODEL_MOL_TAG_STYLE[key]
    if "chemlactica" in key or "galactica" in key or "chemma" in key:
        return "chemlactica"
    if "chemllama" in key or "llama" in key:
        return "chemllama"
    raise ValueError(
        f"Cannot infer mol_tag_style for model_size={model_size!r}. "
        f"Known models: {sorted(MODEL_MOL_TAG_STYLE)}. "
        "Pass --mol_tag_style chemllama|chemlactica explicitly."
    )


def _fmt_prompt_value(v: float) -> str:
    """Two digits after the decimal in prompts (e.g. 0.6620 -> 0.66)."""
    return f"{float(v):.2f}"


def build_qed_prompts(
    rows: list[ZincRow], n: int, seed: int, fewshot_k: int, mol_open: str, mol_close: str
) -> list[dict]:
    import random

    rng = random.Random(seed)
    chosen = rng.sample(rows, k=n)
    prefix = ""
    if fewshot_k > 0:
        demos = random.Random(seed + 10_000).sample(rows, k=fewshot_k)
        prefix = (
            "\n".join(
                f"[QED]{_fmt_prompt_value(float(r.qed))}[/QED]{mol_open}{r.smiles}{mol_close}"
                for r in demos
            )
            + "\n"
        )
    out = []
    for i, r in enumerate(chosen):
        out.append(
            {
                "id": f"qed-{i:04d}",
                "property": "qed",
                "target": float(r.qed),
                "targets": {"qed": float(r.qed)},
                "ref_smiles": None,
                "prompt": prefix + f"[QED]{_fmt_prompt_value(float(r.qed))}[/QED]{mol_open}",
            }
        )
    return out


def build_sas_prompts(
    rows: list[ZincRow], n: int, seed: int, fewshot_k: int, mol_open: str, mol_close: str
) -> list[dict]:
    import random

    rng = random.Random(seed)
    chosen = rng.sample(rows, k=n)
    prefix = ""
    if fewshot_k > 0:
        demos = random.Random(seed + 20_000).sample(rows, k=fewshot_k)
        prefix = (
            "\n".join(
                f"[SAS]{_fmt_prompt_value(float(r.sas))}[/SAS]{mol_open}{r.smiles}{mol_close}"
                for r in demos
            )
            + "\n"
        )
    out = []
    for i, r in enumerate(chosen):
        out.append(
            {
                "id": f"sas-{i:04d}",
                "property": "sas",
                "target": float(r.sas),
                "targets": {"sas": float(r.sas)},
                "ref_smiles": None,
                "prompt": prefix + f"[SAS]{_fmt_prompt_value(float(r.sas))}[/SAS]{mol_open}",
            }
        )
    return out


def build_similarity_random_prompts(smiles: list[str], n: int, seed: int, mol_open: str) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        mol1 = smiles[int(rng.integers(0, len(smiles)))]
        target = float(rng.random())
        out.append(
            {
                "id": f"simr-{i:04d}",
                "property": "similarity_random",
                "target": target,
                "targets": {"similarity": target},
                "ref_smiles": mol1,
                "prompt": f"[SIMILAR]{mol1} {_fmt_prompt_value(target)}[/SIMILAR]{mol_open}",
            }
        )
    return out


def build_qed_sas_prompts(
    rows: list[ZincRow], n: int, seed: int, fewshot_k: int, mol_open: str, mol_close: str
) -> list[dict]:
    """Double conditioning: [QED]x[/QED][SAS]y[/SAS][SMILES]."""
    import random

    rng = random.Random(seed)
    chosen = rng.sample(rows, k=n)
    prefix = ""
    if fewshot_k > 0:
        demos = random.Random(seed + 30_000).sample(rows, k=fewshot_k)
        prefix = (
            "\n".join(
                f"[QED]{_fmt_prompt_value(float(r.qed))}[/QED]"
                f"[SAS]{_fmt_prompt_value(float(r.sas))}[/SAS]"
                f"{mol_open}{r.smiles}{mol_close}"
                for r in demos
            )
            + "\n"
        )
    out = []
    for i, r in enumerate(chosen):
        qed = float(r.qed)
        sas = float(r.sas)
        out.append(
            {
                "id": f"qed_sas-{i:04d}",
                "property": "qed_sas",
                "target": qed,  # primary for backward-compatible CSV tooling
                "targets": {"qed": qed, "sas": sas},
                "ref_smiles": None,
                "prompt": (
                    prefix
                    + f"[QED]{_fmt_prompt_value(qed)}[/QED]"
                    + f"[SAS]{_fmt_prompt_value(sas)}[/SAS]"
                    + mol_open
                ),
            }
        )
    return out


def build_qed_sas_similarity_random_prompts(
    rows: list[ZincRow], n: int, seed: int, fewshot_k: int, mol_open: str, mol_close: str
) -> list[dict]:
    """
    Triple conditioning:
      [QED]x[/QED][SAS]y[/SAS][SIMILAR]ref z[/SIMILAR][SMILES]
    QED/SAS from a ZINC molecule; similarity target is random in [0,1] vs a random ref.
    """
    return build_conditioned_prompts(
        rows,
        n=n,
        seed=seed,
        fewshot_k=fewshot_k,
        mol_open=mol_open,
        mol_close=mol_close,
        properties=("qed", "sas", "similarity"),
        task="qed_sas_similarity_random",
        id_prefix="qed_sas_simr",
        fewshot_salt=40_000,
    )


def _property_tags(
    *,
    properties: tuple[str, ...],
    qed: float | None,
    sas: float | None,
    ref: str | None,
    sim: float | None,
) -> str:
    parts: list[str] = []
    for key in properties:
        if key == "qed":
            parts.append(f"[QED]{_fmt_prompt_value(float(qed))}[/QED]")
        elif key == "sas":
            parts.append(f"[SAS]{_fmt_prompt_value(float(sas))}[/SAS]")
        elif key == "similarity":
            parts.append(f"[SIMILAR]{ref} {_fmt_prompt_value(float(sim))}[/SIMILAR]")
        else:
            raise ValueError(f"Unknown property {key}")
    return "".join(parts)


def build_conditioned_prompts(
    rows: list[ZincRow],
    n: int,
    seed: int,
    fewshot_k: int,
    mol_open: str,
    mol_close: str,
    properties: tuple[str, ...],
    task: str,
    id_prefix: str,
    fewshot_salt: int,
) -> list[dict]:
    """Build prompts for a property subset. ``properties`` order is tag order."""
    import random

    rng = random.Random(seed)
    chosen = rng.sample(rows, k=n)
    smiles = [r.smiles for r in rows]
    needs_sim = "similarity" in properties

    def _sample_sim() -> tuple[str, float]:
        return smiles[rng.randrange(len(smiles))], float(rng.random())

    prefix = ""
    if fewshot_k > 0:
        demos = random.Random(seed + fewshot_salt).sample(rows, k=fewshot_k)
        demo_lines = []
        for r in demos:
            ref, sim = _sample_sim() if needs_sim else (None, None)
            tags = _property_tags(
                properties=properties,
                qed=float(r.qed),
                sas=float(r.sas),
                ref=ref,
                sim=sim,
            )
            demo_lines.append(f"{tags}{mol_open}{r.smiles}{mol_close}")
        prefix = "\n".join(demo_lines) + "\n"

    out = []
    for i, r in enumerate(chosen):
        qed = float(r.qed)
        sas = float(r.sas)
        ref, sim = _sample_sim() if needs_sim else (None, None)
        targets: dict[str, float] = {}
        if "qed" in properties:
            targets["qed"] = qed
        if "sas" in properties:
            targets["sas"] = sas
        if "similarity" in properties:
            targets["similarity"] = float(sim)
        primary = next(iter(targets.values()))
        out.append(
            {
                "id": f"{id_prefix}-{i:04d}",
                "property": task,
                "target": primary,
                "targets": targets,
                "ref_smiles": ref,
                "prompt": prefix + _property_tags(
                    properties=properties,
                    qed=qed,
                    sas=sas,
                    ref=ref,
                    sim=sim,
                )
                + mol_open,
            }
        )
    return out


def write_task_prompts(
    *,
    out_dir: Path,
    tasks: Iterable[str],
    n: int,
    seed: int,
    fewshot_k: int,
    mol_tag_style: str,
    zinc_csv: Path | None = None,
) -> dict:
    """
    Build and persist prompt JSONL files for the requested tasks.
    Returns the manifest dict written next to the prompts.
    """
    quiet_rdkit()
    if mol_tag_style not in MOL_TAG_STYLES:
        raise ValueError(f"Unknown mol_tag_style={mol_tag_style!r}")
    mol_open, mol_close = MOL_TAG_STYLES[mol_tag_style]
    csv_path = Path(zinc_csv) if zinc_csv else project_path("genmol", "data", "zinc250k.csv")
    rows = read_zinc250k_rows(csv_path)
    if len(rows) < n:
        raise ValueError(f"Dataset too small: {len(rows)} rows < n={n}")
    smiles = [r.smiles for r in rows]
    ensure_dir(out_dir)

    task_set = list(tasks)
    for task in task_set:
        keys = task_property_keys(task)
        task_seed = seed + TASK_SEED_OFFSET[task]
        if task == "qed":
            prompts = build_qed_prompts(
                rows, n=n, seed=task_seed, fewshot_k=fewshot_k, mol_open=mol_open, mol_close=mol_close
            )
        elif task == "sas":
            prompts = build_sas_prompts(
                rows, n=n, seed=task_seed, fewshot_k=fewshot_k, mol_open=mol_open, mol_close=mol_close
            )
        elif task == "similarity_random":
            prompts = build_similarity_random_prompts(smiles, n=n, seed=task_seed, mol_open=mol_open)
        elif task == "qed_sas":
            prompts = build_qed_sas_prompts(
                rows, n=n, seed=task_seed, fewshot_k=fewshot_k, mol_open=mol_open, mol_close=mol_close
            )
        else:
            prompts = build_conditioned_prompts(
                rows,
                n=n,
                seed=task_seed,
                fewshot_k=fewshot_k,
                mol_open=mol_open,
                mol_close=mol_close,
                properties=keys,
                task=task,
                id_prefix=task.replace("similarity_random", "simr"),
                fewshot_salt=10_000 * (TASK_SEED_OFFSET[task] + 1),
            )
        safe_jsonl_write(out_dir / f"{task}.jsonl", prompts)

    manifest = {
        "n": n,
        "seed": seed,
        "fewshot_k": fewshot_k,
        "mol_tag_style": mol_tag_style,
        "mol_open": mol_open,
        "mol_close": mol_close,
        "tasks": task_set,
        "zinc_csv": str(csv_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def ensure_prompts(
    *,
    prompts_dir: Path,
    tasks: Iterable[str],
    n: int,
    seed: int,
    fewshot_k: int,
    mol_tag_style: str,
    zinc_csv: Path | None = None,
    rebuild: bool = False,
) -> dict:
    """
    Reuse existing prompts when the manifest matches; otherwise build and keep them.

    Shared prompts live in the repo (Cond-Gen/prompts). Refusing a silent rewrite
    when n/seed/style disagree avoids clobbering the canonical files.
    """
    tasks = list(tasks)
    manifest_path = prompts_dir / "manifest.json"
    if not rebuild and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            same_setup = (
                int(manifest.get("n", -1)) == n
                and int(manifest.get("seed", -1)) == seed
                and int(manifest.get("fewshot_k", -1)) == fewshot_k
                and str(manifest.get("mol_tag_style", "")) == mol_tag_style
            )
            if same_setup:
                missing = [t for t in tasks if not (prompts_dir / f"{t}.jsonl").exists()]
                if not missing and set(manifest.get("tasks", [])) >= set(tasks):
                    return manifest
                if missing:
                    extra = write_task_prompts(
                        out_dir=prompts_dir,
                        tasks=missing,
                        n=n,
                        seed=seed,
                        fewshot_k=fewshot_k,
                        mol_tag_style=mol_tag_style,
                        zinc_csv=zinc_csv,
                    )
                    merged = sorted(set(manifest.get("tasks", [])) | set(extra["tasks"]))
                    manifest = dict(extra)
                    manifest["tasks"] = merged
                    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                    return manifest
            raise RuntimeError(
                f"Prompt manifest at {manifest_path} does not match n={n} seed={seed} "
                f"fewshot_k={fewshot_k} mol_tag_style={mol_tag_style} tasks={tasks}. "
                "Pass --rebuild_prompts to overwrite the shared files, or point "
                "--prompts_dir at a different directory."
            )
        except RuntimeError:
            raise
        except Exception:
            pass
    return write_task_prompts(
        out_dir=prompts_dir,
        tasks=tasks,
        n=n,
        seed=seed,
        fewshot_k=fewshot_k,
        mol_tag_style=mol_tag_style,
        zinc_csv=zinc_csv,
    )


def compute_property_for_task(
    task: str, generated_smiles: str | None, ref_smiles: str | None = None
) -> float | None:
    preds = compute_properties_for_task(task, generated_smiles, ref_smiles)
    keys = task_property_keys(task)
    if len(keys) != 1:
        # Multi-property tasks should use compute_properties_for_task.
        raise ValueError(f"Use compute_properties_for_task for multi-property task={task}")
    return preds.get(keys[0])


def compute_properties_for_task(
    task: str, generated_smiles: str | None, ref_smiles: str | None = None
) -> dict[str, float | None]:
    """Compute all properties required by ``task``; missing -> None."""
    keys = task_property_keys(task)
    out: dict[str, float | None] = {k: None for k in keys}
    if not generated_smiles:
        return out
    if "qed" in out:
        out["qed"] = compute_qed(generated_smiles)
    if "sas" in out:
        out["sas"] = compute_sas(generated_smiles)
    if "similarity" in out:
        if not ref_smiles:
            out["similarity"] = None
        else:
            out["similarity"] = compute_similarity(ref_smiles, generated_smiles)
    return out


def row_targets(row: dict, task: str) -> dict[str, float]:
    """Normalize prompt-row targets into a property->float map."""
    if isinstance(row.get("targets"), dict) and row["targets"]:
        return {str(k): float(v) for k, v in row["targets"].items()}
    keys = task_property_keys(task)
    if len(keys) == 1 and row.get("target") is not None:
        return {keys[0]: float(row["target"])}
    raise ValueError(f"Prompt row missing targets for task={task}: {row.get('id')}")


def plot_desired_vs_computed(
    y_true: list[float],
    y_pred: list[float],
    out_png: Path,
    *,
    title: str,
    xlabel: str = "desired property value",
    ylabel: str = "computed property (generated)",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=10, alpha=0.6)
    lo = min(min(y_true), min(y_pred))
    hi = max(max(y_true), max(y_pred))
    plt.plot([lo, hi], [lo, hi], linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_error_hist(errs: list[float], out_png: Path, *, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.hist(errs, bins=40)
    plt.xlabel("error (computed - desired)")
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=200)
    plt.close()


