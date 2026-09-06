"""
Unified Bayesian / Grid Hyperparameter Search for Polaris ADME Benchmark across ChemLlama model scales.
"""
import os
import sys
import argparse
import traceback
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
import wandb
import polaris as po

# Allow importing from sibling utils module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.models import ChemlacticaMultiTaskRegressor
from utils.dataset import SmilesDataset, collate_fn
from utils.curation import (
    _ensure_smiles_column,
    create_stratified_folds,
    get_train_test_dfs
)
from utils.metrics import masked_mse_loss, evaluate_multitask, _set_seed

MODEL_MAP = {
    "170m": "yerevann/ChemLlama-170M",
    "380m": "yerevann/ChemLlama-380M",
    "1b": "yerevann/ChemLlama-1B",
    "1.3b": "yerevann/ChemLlama-1B",
    "3b": "yerevann/ChemLlama-3B"
}

UNFREEZE_CANDIDATES = {
    "170m": [2, 4, 8],
    "380m": [2, 4, 8, 12, 16],
    "1b": [2, 4, 8, 12, 16],
    "1.3b": [2, 4, 8, 12, 16],
    "3b": [2, 4, 8, 12, 16, 20, 24, 28]
}

POLARIS_TO_INTERNAL = {
    "LOG_HLM_CLint": "HLM",
    "LOG_SOLUBILITY": "Sol",
    "LOG_MDR1-MDCK_ER": "MDR1",
    "LOG_RLM_CLint": "RLM",
    "LOG_HPPB": "hPPB",
    "LOG_RPPB": "rPPB"
}
INTERNAL_TASKS = list(POLARIS_TO_INTERNAL.values())


def train_trial(train_split, val_split, tokenizer, device, config, model_name, smiles_col):
    ds_train = SmilesDataset(train_split, tokenizer, max_len=128, smiles_col=smiles_col, tasks=INTERNAL_TASKS)
    ds_val = SmilesDataset(val_split, tokenizer, max_len=128, smiles_col=smiles_col, tasks=INTERNAL_TASKS)

    train_loader = DataLoader(
        ds_train,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda")
    )

    model = ChemlacticaMultiTaskRegressor(
        model_name=model_name,
        tasks=INTERNAL_TASKS,
        pooling=config["pooling"],
        mlp_hidden_size=config["mlp_hidden_size"],
        mlp_layers=config["mlp_layers"],
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
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = []
            for batch in train_loader:
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
                running_loss.append(loss.item())

            train_mse = float(np.mean(running_loss))
            val_metrics = evaluate_multitask(model, val_loader, device=device, tasks=INTERNAL_TASKS)
            val_mean_r2 = val_metrics["_mean_r2"]
            val_mean_mse = float(np.nanmean([val_metrics[t]["mse"] for t in INTERNAL_TASKS]))

            log_dict = {
                "epoch": epoch,
                "train/loss_mse": train_mse,
                "val/mean_r2": val_mean_r2,
                "val/mean_mse": val_mean_mse,
                "lr/head": scheduler.get_last_lr()[-1]
            }
            for t in INTERNAL_TASKS:
                log_dict[f"val/{t}_r2"] = val_metrics[t]["r2"]
                log_dict[f"val/{t}_mse"] = val_metrics[t]["mse"]
            wandb.log(log_dict)
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
    parser = argparse.ArgumentParser(description="Polaris ADME Bayesian / Grid Search Sweep")
    parser.add_argument("--model", type=str, default="170m", help="Model scale (170m, 380m, 1b, 3b) or Hugging Face repository ID")
    parser.add_argument("--benchmark", type=str, default="polaris/adme-fang-r-1", help="Polaris benchmark identifier")
    parser.add_argument("--project", type=str, default="chemlactica-adme-sweeps", help="WandB project name")
    parser.add_argument("--entity", type=str, default=None, help="WandB entity name")
    parser.add_argument("--count", type=int, default=100, help="Number of sweep trials to execute")
    parser.add_argument("--sweep_id", type=str, default=None, help="Existing WandB sweep ID to resume")
    args = parser.parse_args()

    model_key = args.model.lower()
    checkpoint_path = MODEL_MAP.get(model_key, args.model)
    unfreeze_vals = UNFREEZE_CANDIDATES.get(model_key, [2, 4, 8, 16])
    batch_sizes = [16, 32] if model_key == "3b" else [16, 32, 64]

    print(f"Initializing Polaris ADME Sweep for Model: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["[START_SMILES]", "[END_SMILES]"]})

    # Load Polaris benchmark and construct stratified folds
    benchmark = po.load_benchmark(args.benchmark)
    train_df, _ = get_train_test_dfs(benchmark, rename_targets=POLARIS_TO_INTERNAL, test_same_shape_as_train=True)
    train_df2, smiles_col = _ensure_smiles_column(train_df.copy(), benchmark)

    print("Generating Stratified Folds on Polaris training data...")
    folds = create_stratified_folds(train_df2, INTERNAL_TASKS, n_splits=5)
    train_split = folds[0]["train_df"]
    val_split = folds[0]["val_df"]
    print(f"Fold 0: Train={len(train_split)} molecules, Val={len(val_split)} molecules.")

    sweep_config = {
        "method": "bayes",
        "name": f"polaris-{model_key}-sweep",
        "metric": {"goal": "maximize", "name": "val/mean_r2"},
        "parameters": {
            "head_lr": {"distribution": "log_uniform_values", "min": 1e-5, "max": 5e-2},
            "backbone_lr": {"distribution": "log_uniform_values", "min": 1e-6, "max": 5e-3},
            "weight_decay": {"values": [0.001, 0.05, 0.1, 0.15, 0.2, 0.25]},
            "mlp_layers": {"values": [2]},
            "mlp_hidden_size": {"values": [256, 512]},
            "pooling": {"values": ["mean", "last_token", "attn"]},
            "n_unfreeze": {"values": unfreeze_vals},
            "batch_size": {"values": batch_sizes}
        }
    }

    sweep_id = args.sweep_id
    if sweep_id is None:
        sweep_id = wandb.sweep(sweep_config, project=args.project, entity=args.entity)
        print(f"Created new WandB Sweep: {sweep_id}")
    else:
        print(f"Resuming existing WandB Sweep: {sweep_id}")

    def sweep_train():
        with wandb.init() as run:
            _set_seed(42)
            config = {
                "head_lr": wandb.config.head_lr,
                "backbone_lr": wandb.config.backbone_lr,
                "weight_decay": wandb.config.weight_decay,
                "mlp_layers": wandb.config.mlp_layers,
                "mlp_hidden_size": wandb.config.mlp_hidden_size,
                "pooling": wandb.config.pooling,
                "n_unfreeze": wandb.config.n_unfreeze,
                "batch_size": wandb.config.batch_size,
                "dropout": 0.1,
                "epochs": 10
            }

            try:
                train_trial(train_split, val_split, tokenizer, device, config, checkpoint_path, smiles_col)
            except Exception as e:
                print(f"Trial failed with exception: {e}")
                traceback.print_exc()
                traceback.clear_frames(e.__traceback__)
            finally:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

    wandb.agent(sweep_id, project=args.project, entity=args.entity, function=sweep_train, count=args.count)
    print("Sweep sequence completed.")


if __name__ == "__main__":
    main()
