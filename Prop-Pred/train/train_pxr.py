"""
Multi-Seed Full Dataset Training for PXR Regression across ChemLlama scales.
"""
import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

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
from utils.metrics import masked_mae_loss, evaluate_regression, _set_seed


def train_single_seed(train_df, tokenizer, device, config, seed, best_weight_path, final_weight_path):
    print(f"\n==========================================")
    print(f"Starting PXR Training for Seed {seed}")
    print(f"==========================================")
    _set_seed(seed)

    ds_train = SmilesDataset(train_df, tokenizer, max_len=128, smiles_col="smiles", target_col="pEC50")
    train_loader = DataLoader(
        ds_train,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        ds_train,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda")
    )

    model = ChemlacticaRegressor(
        model_name=config["model_name"],
        pooling=config["pooling"],
        mlp_hidden_size=config["mlp_hidden_size"],
        mlp_layers=config.get("mlp_layers", 2),
        dropout=config.get("dropout", 0.1),
        unfreeze_last_n=config["n_unfreeze"],
        tokenizer_len=len(tokenizer)
    ).to(device)

    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head") and not n.startswith("shared_mlp")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and (n.startswith("head") or n.startswith("shared_mlp"))]

    param_groups = []
    if len(backbone_params) > 0:
        param_groups.append({"params": backbone_params, "lr": config["backbone_lr"], "weight_decay": config["weight_decay"]})
    if len(head_params) > 0:
        param_groups.append({"params": head_params, "lr": config["head_lr"], "weight_decay": config["weight_decay"]})

    optimizer = torch.optim.AdamW(param_groups)
    epochs = config.get("epochs", 10)
    total_steps = epochs * len(train_loader)
    warmup_steps = int(config.get("warmup_ratio", 0.1) * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_loss = float("inf")
    best_state = None

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            running = []
            for batch in tqdm(train_loader, desc=f"Seed {seed} Epoch {epoch}/{epochs}", leave=False):
                optimizer.zero_grad(set_to_none=True)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["targets"].to(device)
                masks = batch["masks"].to(device)
                weights = batch["weights"].to(device)

                preds = model(input_ids=input_ids, attention_mask=attention_mask)
                loss, _ = masked_mae_loss(preds, targets, masks, weights=weights)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                running.append(loss.item())

            train_mae = float(np.mean(running))
            metrics = evaluate_regression(model, val_loader, device=device)
            val_rae, val_r2 = metrics["rae"], metrics["r2"]
            print(f"[Seed {seed} Epoch {epoch}] Train MAE: {train_mae:.4f} | Val RAE: {val_rae:.4f} | Val R2: {val_r2:.4f}")

            if val_rae < best_loss:
                best_loss = val_rae
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # Save weights
        os.makedirs(os.path.dirname(best_weight_path), exist_ok=True)
        if best_state is not None:
            torch.save({"state_dict": best_state}, best_weight_path)
            print(f"Saved best weights: {best_weight_path}")

        final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save({"state_dict": final_state}, final_weight_path)
        print(f"Saved final weights: {final_weight_path}")
    finally:
        del model
        del optimizer
        del scheduler
        del train_loader
        del val_loader
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Training for PXR Regression")
    parser.add_argument("--model", type=str, default="380m", help="Model scale (170m, 380m, 1b, 3b)")
    parser.add_argument("--config", type=str, default=None, help="Path to custom config YAML")
    default_data = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "pxr", "pxr_train.csv"))
    parser.add_argument("--data_path", type=str, default=default_data, help="Path to curated training dataset")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 81, 75, 95, 114], help="List of random seeds")
    parser.add_argument("--output_dir", type=str, default="./weights/pxr", help="Directory to save model checkpoints")
    args = parser.parse_args()

    model_key = args.model.lower()
    config_path = args.config if args.config else os.path.join(os.path.dirname(__file__), "configs", "pxr.yaml")

    with open(config_path, "r") as f:
        all_configs = yaml.safe_load(f)

    if model_key not in all_configs:
        raise ValueError(f"Model key '{model_key}' not found in {config_path}. Available: {list(all_configs.keys())}")

    config = all_configs[model_key]
    model_name = config["model_name"]
    print(f"Configuring PXR Multi-Seed Training for {model_key.upper()} ({model_name})")
    print(f"Seeds: {args.seeds}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["[START_SMILES]", "[END_SMILES]"]})

    # Curation pipeline
    print(f"Loading and curating training dataset from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    df = standardize_tautomers_and_smiles(df, smiles_col_in="SMILES" if "SMILES" in df.columns else "smiles")
    df = exclude_reactive_electrophiles(df, smiles_col="smiles")
    df = compute_sample_weights(df, smiles_col="smiles")
    df, _ = _ensure_smiles_column(df)
    print(f"Curated full training dataset size: {len(df)} molecules.")

    save_dir = os.path.join(args.output_dir, model_key)
    for seed in args.seeds:
        best_path = os.path.join(save_dir, f"pxr_{model_key}_seed_{seed}_best.pt")
        final_path = os.path.join(save_dir, f"pxr_{model_key}_seed_{seed}_final.pt")
        train_single_seed(df, tokenizer, device, config, seed, best_path, final_path)

    print("\nAll seeds completed successfully.")


if __name__ == "__main__":
    main()
