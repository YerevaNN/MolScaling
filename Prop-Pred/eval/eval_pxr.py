"""
Multi-Seed Evaluation and Variance Statistics for PXR Regression.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Allow importing from sibling utils module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.models import ChemlacticaRegressor
from utils.dataset import SmilesDataset, collate_fn
from utils.curation import (
    _ensure_smiles_column,
    standardize_tautomers_and_smiles,
    exclude_reactive_electrophiles,
    compute_sample_weights
)
from utils.metrics import evaluate_regression

MODEL_MAP = {
    "170m": "yerevann/chemllama-170m",
    "380m": "yerevann/ChemLlama-380M",
    "1b": "yerevann/ChemLlama-1B",
    "1.3b": "yerevann/ChemLlama-1B",
    "3b": "yerevann/ChemLlama-3B"
}

ARCH_DEFAULTS = {
    "170m": {"pooling": "last_token", "mlp_hidden_size": 512, "n_unfreeze": 8},
    "380m": {"pooling": "last_token", "mlp_hidden_size": 512, "n_unfreeze": 16},
    "1b": {"pooling": "last_token", "mlp_hidden_size": 256, "n_unfreeze": 12},
    "3b": {"pooling": "attn", "mlp_hidden_size": 256, "n_unfreeze": 24}
}


def main():
    parser = argparse.ArgumentParser(description="Evaluate PXR Multi-Seed Models on Test Set")
    parser.add_argument("--model", type=str, default="380m", help="Model scale (170m, 380m, 1b, 3b)")
    default_test = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "pxr", "pxr_test.csv"))
    parser.add_argument("--test_data", type=str, default=default_test, help="Path to unblinded test CSV")
    parser.add_argument("--weights_dir", type=str, default=None, help="Directory containing seed checkpoints")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 81, 75, 95, 114], help="Seeds to evaluate")
    parser.add_argument("--output_csv", type=str, default=None, help="Path to save variance statistics CSV")
    args = parser.parse_args()

    model_key = args.model.lower()
    model_name = MODEL_MAP.get(model_key, args.model)
    arch = ARCH_DEFAULTS.get(model_key, {"pooling": "last_token", "mlp_hidden_size": 512, "n_unfreeze": 16})

    weights_dir = args.weights_dir if args.weights_dir else f"./weights/pxr/{model_key}"
    output_csv = args.output_csv if args.output_csv else f"./results/pxr_{model_key}_variance_metrics.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["[START_SMILES]", "[END_SMILES]"]})

    print(f"Loading unblinded test dataset from {args.test_data}...")
    df_test = pd.read_csv(args.test_data)
    df_test = standardize_tautomers_and_smiles(df_test, smiles_col_in="SMILES" if "SMILES" in df_test.columns else "smiles")
    df_test = exclude_reactive_electrophiles(df_test, smiles_col="smiles")
    df_test = compute_sample_weights(df_test, smiles_col="smiles")
    df_test, _ = _ensure_smiles_column(df_test)
    print(f"Test dataset size: {len(df_test)} molecules.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_test = SmilesDataset(df_test, tokenizer, max_len=128, smiles_col="smiles", target_col="pEC50")
    test_loader = DataLoader(ds_test, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_fn)

    results = []
    for seed in args.seeds:
        weight_path = os.path.join(weights_dir, f"pxr_{model_key}_seed_{seed}_best.pt")
        if not os.path.exists(weight_path):
            # Also check alternative naming
            alt_path = os.path.join(weights_dir, f"pxr_3_seed_{seed}_best2.pt")
            if os.path.exists(alt_path):
                weight_path = alt_path
            else:
                print(f"Warning: Checkpoint for seed {seed} not found at {weight_path}. Skipping.")
                continue

        print(f"Loading checkpoint for Seed {seed}: {weight_path}")
        model = ChemlacticaRegressor(
            model_name=model_name,
            pooling=arch["pooling"],
            mlp_hidden_size=arch["mlp_hidden_size"],
            mlp_layers=2,
            dropout=0.1,
            unfreeze_last_n=arch["n_unfreeze"],
            tokenizer_len=len(tokenizer)
        ).to(device)

        sd = torch.load(weight_path, map_location=device)
        if "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd)

        metrics = evaluate_regression(model, test_loader, device=device)
        print(f"Seed {seed} Test Metrics: RAE = {metrics['rae']:.4f} | R2 = {metrics['r2']:.4f}")
        results.append({"seed": seed, "rae": metrics["rae"], "r2": metrics["r2"]})

        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(results) == 0:
        print("Error: No checkpoints were evaluated.")
        return

    # Statistical calculations (Student's t critical value for N=5, df=4, 95% CI: 2.7764)
    t_critical = 2.7764 if len(results) == 5 else 2.571
    raes = [r["rae"] for r in results]
    r2s = [r["r2"] for r in results]

    rae_mean, rae_std = float(np.mean(raes)), float(np.std(raes, ddof=1)) if len(raes) > 1 else 0.0
    r2_mean, r2_std = float(np.mean(r2s)), float(np.std(r2s, ddof=1)) if len(r2s) > 1 else 0.0

    rae_margin = t_critical * (rae_std / np.sqrt(len(raes))) if len(raes) > 1 else 0.0
    r2_margin = t_critical * (r2_std / np.sqrt(len(r2s))) if len(r2s) > 1 else 0.0

    print("\n==========================================")
    print(f"PXR {model_key.upper()} Final Evaluation Results (N={len(results)} seeds)")
    print("==========================================")
    print(f"RAE: {rae_mean:.4f} ± {rae_std:.4f}  [95% CI: {rae_mean - rae_margin:.4f} - {rae_mean + rae_margin:.4f}]")
    print(f"R2:  {r2_mean:.4f} ± {r2_std:.4f}  [95% CI: {r2_mean - r2_margin:.4f} - {r2_mean + r2_margin:.4f}]")

    df_out = pd.DataFrame([{
        "Model": model_key.upper(),
        "Task": "pEC50",
        "Num_Seeds": len(results),
        "RAE_mean": rae_mean,
        "RAE_std": rae_std,
        "RAE_CI_lower": rae_mean - rae_margin,
        "RAE_CI_upper": rae_mean + rae_margin,
        "R2_mean": r2_mean,
        "R2_std": r2_std,
        "R2_CI_lower": r2_mean - r2_margin,
        "R2_CI_upper": r2_mean + r2_margin
    }])
    df_out.to_csv(output_csv, index=False)
    print(f"Saved variance metrics to {output_csv}")


if __name__ == "__main__":
    main()
