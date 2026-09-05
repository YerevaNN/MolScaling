"""
Multi-Seed Full Dataset Training for Polaris ADME Benchmark across ChemLlama scales.
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
import polaris as po

# Allow importing from sibling utils module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.models import ChemlacticaMultiTaskRegressor
from utils.dataset import SmilesDataset, collate_fn
from utils.curation import _ensure_smiles_column, get_train_test_dfs
from utils.metrics import masked_mse_loss, evaluate_multitask, _set_seed

POLARIS_TO_INTERNAL = {
    "LOG_HLM_CLint": "HLM",
    "LOG_SOLUBILITY": "Sol",
    "LOG_MDR1-MDCK_ER": "MDR1",
    "LOG_RLM_CLint": "RLM",
    "LOG_HPPB": "hPPB",
    "LOG_RPPB": "rPPB"
}
INTERNAL_TASKS = list(POLARIS_TO_INTERNAL.values())


def train_single_seed(train_df, smiles_col, tokenizer, device, config, seed, best_weight_path, final_weight_path):
    print(f"\n==========================================")
    print(f"Starting Polaris Training for Seed {seed}")
    print(f"==========================================")
    _set_seed(seed)

    ds_train = SmilesDataset(train_df, tokenizer, max_len=128, smiles_col=smiles_col, tasks=INTERNAL_TASKS)
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

    model = ChemlacticaMultiTaskRegressor(
        model_name=config["model_name"],
        tasks=INTERNAL_TASKS,
        pooling=config["pooling"],
        mlp_hidden_size=config["mlp_hidden_size"],
        mlp_layers=config.get("mlp_layers", 2),
        dropout=config.get("dropout", 0.1),
        unfreeze_last_n=config["n_unfreeze"],
        tokenizer_len=len(tokenizer)
    ).to(device)

    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("heads.") and not n.startswith("shared_mlp")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and (n.startswith("heads.") or n.startswith("shared_mlp"))]

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

                preds = model(input_ids=input_ids, attention_mask=attention_mask)
                loss, _ = masked_mse_loss(preds, targets, masks, tasks=INTERNAL_TASKS)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                running.append(loss.item())

            train_mse = float(np.mean(running))
            metrics = evaluate_multitask(model, val_loader, device=device, tasks=INTERNAL_TASKS)
            val_mean_r2 = metrics["_mean_r2"]
            val_mean_mse = float(np.nanmean([metrics[t]["mse"] for t in INTERNAL_TASKS]))
            print(f"[Seed {seed} Epoch {epoch}] Train MSE: {train_mse:.4f} | Val MSE: {val_mean_mse:.4f} | Mean R2: {val_mean_r2:.4f}")

            if val_mean_mse < best_loss:
                best_loss = val_mean_mse
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
    parser = argparse.ArgumentParser(description="Multi-Seed Training for Polaris ADME")
    parser.add_argument("--model", type=str, default="3b", help="Model scale (170m, 380m, 1b, 3b)")
    parser.add_argument("--config", type=str, default=None, help="Path to custom config YAML")
    parser.add_argument("--benchmark", type=str, default="polaris/adme-fang-r-1", help="Polaris benchmark identifier")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 81, 75, 95, 114], help="List of random seeds")
    parser.add_argument("--output_dir", type=str, default="./weights/polaris", help="Directory to save model checkpoints")
    args = parser.parse_args()

    model_key = args.model.lower()
    config_path = args.config if args.config else os.path.join(os.path.dirname(__file__), "configs", "polaris.yaml")

    with open(config_path, "r") as f:
        all_configs = yaml.safe_load(f)

    if model_key not in all_configs:
        raise ValueError(f"Model key '{model_key}' not found in {config_path}. Available: {list(all_configs.keys())}")

    config = all_configs[model_key]
    model_name = config["model_name"]
    print(f"Configuring Polaris Multi-Seed Training for {model_key.upper()} ({model_name})")
    print(f"Seeds: {args.seeds}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["[START_SMILES]", "[END_SMILES]"]})

    # Prepare Polaris training data (100% training split)
    print(f"Loading Polaris benchmark: {args.benchmark}...")
    benchmark = po.load_benchmark(args.benchmark)
    train_df, _ = get_train_test_dfs(benchmark, rename_targets=POLARIS_TO_INTERNAL, test_same_shape_as_train=True)
    train_df2, smiles_col = _ensure_smiles_column(train_df.copy(), benchmark)
    print(f"Full training dataset size: {len(train_df2)} molecules.")

    save_dir = os.path.join(args.output_dir, model_key)
    for seed in args.seeds:
        best_path = os.path.join(save_dir, f"polaris_{model_key}_seed_{seed}_best.pt")
        final_path = os.path.join(save_dir, f"polaris_{model_key}_seed_{seed}_final.pt")
        train_single_seed(train_df2, smiles_col, tokenizer, device, config, seed, best_path, final_path)

    print("\nAll seeds completed successfully.")


if __name__ == "__main__":
    main()
