#!/usr/bin/env python3
"""
Cross-machine identity probe for ChemLlama-170M Cond-Gen collapse.

Loads the Hub model (unless HF_LOCAL_* is set), prints comparable fingerprints,
runs N QED generations, prints validity + a few samples.

On the other machine: run the same script with the same seed and compare
everything under the === markers.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import socket
import sys
from pathlib import Path

import torch

# Cond-Gen helpers
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cond_gen_utils import (  # noqa: E402
    compute_qed,
    configure_tokenizer_for_generation,
    extract_smiles,
    load_hf_model_and_tokenizer,
    read_zinc250k_rows,
    tokenize_prompts_for_generation,
)


def sha256_f32(t: torch.Tensor) -> str:
    x = t.detach().float().cpu().contiguous().view(-1).numpy()
    return hashlib.sha256(x.tobytes()).hexdigest()


def print_section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def main() -> None:
    n = int(os.environ.get("N", "100"))
    seed = int(os.environ.get("SEED", "1"))
    batch_size = int(os.environ.get("BATCH_SIZE", "32"))
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "500"))
    temperature = float(os.environ.get("TEMPERATURE", "1.0"))
    repetition_penalty = float(os.environ.get("REPETITION_PENALTY", "1.01"))
    do_sample = bool(int(os.environ.get("DO_SAMPLE", "1")))
    dtype = os.environ.get("DTYPE", "fp32")
    device = os.environ.get("DEVICE", "cuda")

    # Prefer Hub unless HF_LOCAL_MODEL_DIR is explicitly set.
    hub_id = "yerevann/ChemLlama-170M"
    checkpoint = os.environ.get("HF_LOCAL_MODEL_DIR", "").strip() or hub_id
    tokenizer_path = os.environ.get("HF_LOCAL_TOKENIZER_DIR", "").strip() or checkpoint

    print_section("ENV")
    print(f"hostname={socket.gethostname()}")
    print(f"platform={platform.platform()}")
    print(f"python={sys.version.split()[0]}")
    print(f"torch={torch.__version__}")
    try:
        import transformers

        print(f"transformers={transformers.__version__}")
    except Exception as e:
        print(f"transformers=ERROR {e}")
    try:
        import rdkit

        print(f"rdkit={getattr(rdkit, '__version__', '?')}")
    except Exception as e:
        print(f"rdkit=ERROR {e}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device={torch.cuda.get_device_name(0)}")
    print(f"HF_HOME={os.environ.get('HF_HOME')}")
    print(f"HF_LOCAL_MODEL_DIR={os.environ.get('HF_LOCAL_MODEL_DIR')}")
    print(f"HF_LOCAL_TOKENIZER_DIR={os.environ.get('HF_LOCAL_TOKENIZER_DIR')}")
    print(f"checkpoint={checkpoint}")
    print(f"tokenizer_path={tokenizer_path}")

    print_section("LOAD")
    model, tokenizer = load_hf_model_and_tokenizer(
        checkpoint_path=checkpoint,
        tokenizer_path=tokenizer_path,
        device=device,
        dtype=dtype,
        padding_side="left",
    )
    configure_tokenizer_for_generation(tokenizer)
    p0 = next(model.parameters())
    print(f"param_dtype={p0.dtype} param_device={p0.device}")
    print(f"num_parameters={sum(p.numel() for p in model.parameters())}")

    print_section("TOKENIZER")
    print(f"name_or_path={getattr(tokenizer, 'name_or_path', None)}")
    print(f"bos_token_id={tokenizer.bos_token_id} eos_token_id={tokenizer.eos_token_id} pad_token_id={tokenizer.pad_token_id}")
    print(f"add_bos_token={getattr(tokenizer, 'add_bos_token', None)} padding_side={tokenizer.padding_side}")
    try:
        eot = tokenizer.encode("<|end_of_text|>", add_special_tokens=False)
        print(f"end_of_text_ids={eot}")
        eos_for_gen = int(eot[0]) if len(eot) == 1 else tokenizer.eos_token_id
    except Exception:
        eos_for_gen = tokenizer.eos_token_id
        print("end_of_text_ids=UNAVAILABLE")
    print(f"eos_token_id_used_in_generate={eos_for_gen}")

    sample_prompt = "[QED]0.78[/QED][SMILES]"
    sample_ids = tokenize_prompts_for_generation(tokenizer, [sample_prompt], "cpu")["input_ids"][0].tolist()
    print(f"sample_prompt={sample_prompt!r}")
    print(f"sample_ids={sample_ids}")
    print(f"sample_decode={tokenizer.decode(sample_ids, skip_special_tokens=False)!r}")

    print_section("WEIGHT_FINGERPRINTS")
    named = dict(model.named_parameters())
    for name in [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "lm_head.weight",
    ]:
        if name not in named:
            print(f"{name}=MISSING")
            continue
        t = named[name].data
        x = t.detach().float().cpu().view(-1)
        print(
            f"{name} shape={list(t.shape)} dtype={t.dtype} "
            f"sha256_f32={sha256_f32(t)} "
            f"mean={float(x.mean()):.8f} std={float(x.std(unbiased=False)):.8f} "
            f"head4={[float(v) for v in x[:4].tolist()]} "
            f"tail4={[float(v) for v in x[-4:].tolist()]}"
        )

    # File hashes if local dir
    print_section("CHECKPOINT_FILES")
    ckpt = Path(checkpoint)
    if ckpt.is_dir():
        for fname in ["config.json", "tokenizer.json", "model.safetensors"]:
            fp = ckpt / fname
            if not fp.exists():
                print(f"{fname}=MISSING")
                continue
            h = hashlib.sha256()
            with fp.open("rb") as f:
                while True:
                    b = f.read(8 * 1024 * 1024)
                    if not b:
                        break
                    h.update(b)
            print(f"{fname} size={fp.stat().st_size} sha256={h.hexdigest()}")
    else:
        print(f"checkpoint_is_hub_id={checkpoint}")

    print_section("GREEDY_PROBE")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    toks = tokenize_prompts_for_generation(tokenizer, [sample_prompt], device)
    with torch.inference_mode():
        gen = model.generate(
            **toks,
            max_new_tokens=64,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_for_gen,
        )
    prompt_len = int(toks["attention_mask"][0].sum().item())
    new_ids = gen[0, prompt_len:].tolist()
    text = tokenizer.decode(gen[0].tolist(), skip_special_tokens=False)
    smi = extract_smiles(text, similarity=False)
    print(f"greedy_new_token_ids={new_ids}")
    print(f"greedy_extracted_smiles={smi!r}")
    print(f"greedy_text_prefix={text[:300]!r}")

    # Build 100 QED prompts (same scheme as Cond-Gen)
    print_section("GEN_SETTINGS")
    print(
        f"n={n} seed={seed} batch_size={batch_size} max_new_tokens={max_new_tokens} "
        f"temperature={temperature} repetition_penalty={repetition_penalty} do_sample={do_sample} dtype={dtype}"
    )

    project_root = Path(os.environ.get("PROJECT_ROOT", ROOT.parent))
    zinc = project_root / "genmol" / "data" / "zinc250k.csv"
    rows = read_zinc250k_rows(zinc)
    rng = random.Random(seed)
    chosen = rng.sample(rows, k=n)
    prompts = [f"[QED]{float(r.qed):.2f}[/QED][SMILES]" for r in chosen]
    targets = [float(r.qed) for r in chosen]
    print(f"first_prompt={prompts[0]!r} target={targets[0]}")
    print(f"first_prompt_ids={tokenize_prompts_for_generation(tokenizer, [prompts[0]], 'cpu')['input_ids'][0].tolist()}")

    print_section("GENERATION")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    valid = 0
    invalid = 0
    abs_errs = []
    samples_out = []
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            batch_prompts = prompts[start : start + batch_size]
            batch_targets = targets[start : start + batch_size]
            toks = tokenize_prompts_for_generation(tokenizer, batch_prompts, device)
            gen = model.generate(
                **toks,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_for_gen,
            )
            texts = tokenizer.batch_decode(gen, skip_special_tokens=False)
            for i, (txt, tgt) in enumerate(zip(texts, batch_targets, strict=True)):
                smi = extract_smiles(txt, similarity=False)
                pred = compute_qed(smi) if smi else None
                if pred is None:
                    invalid += 1
                else:
                    valid += 1
                    abs_errs.append(abs(pred - tgt))
                if start + i < 5 or (pred is None and len(samples_out) < 8):
                    samples_out.append(
                        {
                            "i": start + i,
                            "target": tgt,
                            "prompt": batch_prompts[i],
                            "smiles": smi,
                            "pred": pred,
                            "text_prefix": txt[:200],
                        }
                    )
            # Progress + live weight peek (prove same tensors during gen)
            emb = named["model.embed_tokens.weight"].data
            print(
                f"progress={min(start + batch_size, n)}/{n} "
                f"valid_so_far={valid} invalid_so_far={invalid} "
                f"embed_sha256_f32={sha256_f32(emb)} "
                f"embed_mean={float(emb.detach().float().mean()):.8f}",
                flush=True,
            )

    mae = (sum(abs_errs) / len(abs_errs)) if abs_errs else float("nan")
    print_section("RESULTS")
    print(f"valid={valid}/{n} valid_rate={valid / n:.4f} invalid={invalid} mae={mae:.6f}")
    print("samples:")
    for s in samples_out[:8]:
        print(json.dumps(s, ensure_ascii=False))

    print_section("COMPARE_CHECKLIST")
    print("Copy/compare these lines with the other machine:")
    print("1) torch / transformers / rdkit versions")
    print("2) WEIGHT_FINGERPRINTS sha256_f32 for all 4 tensors")
    print("3) sample_ids for [QED]0.78[/QED][SMILES]")
    print("4) greedy_new_token_ids + greedy_extracted_smiles")
    print("5) eos_token_id_used_in_generate + pad_token_id")
    print("6) RESULTS valid_rate / mae")
    print("If 2+3+4 match and valid_rate still differs → sampling / RDKit / env noise.")
    print("If 2 differs → different weights actually loaded.")
    print("If 3 or 5 differ → tokenizer / generate kwargs differ.")


if __name__ == "__main__":
    main()
