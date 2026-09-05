"""
Unified Bayesian / Grid Hyperparameter Search for Belka Multi-target Classification across ChemLlama model scales.
"""
import os
import sys
import glob
import random
import hashlib
import argparse
import traceback
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torchmetrics.classification import MultilabelAveragePrecision
from rdkit import Chem
import wandb

# Allow importing from sibling utils module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.models import _freeze_all_but_last_n, AttnPool, _mean_pool
from utils.metrics import _set_seed

MODEL_MAP = {
    "170m": "yerevann/chemllama-170m",
    "380m": "yerevann/ChemLlama-380M",
    "1b": "yerevann/ChemLlama-1B",
    "1.3b": "yerevann/ChemLlama-1B",
    "3b": "yerevann/ChemLlama-3B"
}

UNFREEZE_CANDIDATES = {
    "170m": [4, 8],
    "380m": [4, 8, 16],
    "1b": [4, 8, 16],
    "1.3b": [4, 8, 16],
    "3b": [4, 8, 16, 24]
}


class BelkaTrainDataset(IterableDataset):
    def __init__(self, tokenizer, data_dir: str, max_len: int = 256, seed: int = 42):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.max_len = max_len
        self.seed = seed
        self.epoch = 0
        self.rank = int(os.environ.get("RANK", 0))
        self.world = int(os.environ.get("WORLD_SIZE", 1))

    def set_epoch(self, e: int):
        self.epoch = e

    def __iter__(self):
        files = sorted(glob.glob(os.path.join(self.data_dir, "*.parquet")))
        files = files[self.rank :: self.world]
        wi = torch.utils.data.get_worker_info()
        worker_id = wi.id if wi else 0
        n_workers = wi.num_workers if wi else 1
        files = files[worker_id :: n_workers]

        if not files:
            return

        rng = random.Random((self.seed + self.epoch * 10000 + worker_id) % 2**32)
        while True:
            rng.shuffle(files)
            for f in files:
                pf = pq.ParquetFile(f)
                groups = list(range(pf.num_row_groups))
                rng.shuffle(groups)
                for g in groups:
                    t = pf.read_row_group(g)
                    sm = t["smiles"].to_numpy()
                    sub = t["subset"].to_numpy()
                    idx = [
                        i for i, s in enumerate(sm)
                        if sub[i] == 0 and int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % 10 == 0
                    ]
                    if len(idx) == 0:
                        continue
                    b4 = t["BRD4"].to_numpy()
                    hs = t["HSA"].to_numpy()
                    se = t["sEH"].to_numpy()
                    rng.shuffle(idx)
                    for i in idx:
                        yield self._enc(sm[i], b4[i], hs[i], se[i])

    def _enc(self, s, a, b, c):
        prompt = f"[START_SMILES]{s}[END_SMILES]"
        e = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": e["input_ids"].squeeze(0),
            "attention_mask": e["attention_mask"].squeeze(0),
            "labels": torch.tensor([a, b, c], dtype=torch.float)
        }


class BelkaValDataset(Dataset):
    def __init__(self, tokenizer, data_dir: str, max_len: int = 256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.rows = []
        print(f"Loading Belka validation set from {data_dir}...")
        for f in sorted(glob.glob(os.path.join(data_dir, "*.parquet"))):
            t = pq.read_table(f)
            sm_list = t["smiles"].to_pylist()
            b4_list = t["BRD4"].to_pylist()
            hs_list = t["HSA"].to_pylist()
            se_list = t["sEH"].to_pylist()
            sub_list = t["subset"].to_pylist()

            for i, s_str in enumerate(sm_list):
                if sub_list[i] == 1 and int(hashlib.md5(s_str.encode("utf-8")).hexdigest(), 16) % 10 == 0:
                    self.rows.append((s_str, b4_list[i], hs_list[i], se_list[i]))
        print(f"Loaded {len(self.rows)} validation samples.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        s, a, b, c = self.rows[idx]
        prompt = f"[START_SMILES]{s}[END_SMILES]"
        e = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": e["input_ids"].squeeze(0),
            "attention_mask": e["attention_mask"].squeeze(0),
            "labels": torch.tensor([a, b, c], dtype=torch.float)
        }


class BelkaClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        pooling: str = "last_token",
        mlp_hidden_size: int = 512,
        dropout: float = 0.25,
        unfreeze_last_n: int = 4,
        tokenizer_len: int = 128258,
        num_targets: int = 3
    ):
        super().__init__()
        self.pooling = pooling
        token = os.environ.get("HF_TOKEN")
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name, token=token)
        self.backbone.config.use_cache = False
        self.backbone.resize_token_embeddings(tokenizer_len)
        _freeze_all_but_last_n(self.backbone, unfreeze_last_n=unfreeze_last_n)

        is_3b = "3B" in model_name or "3b" in model_name
        if is_3b:
            if hasattr(self.backbone, "gradient_checkpointing_enable"):
                self.backbone.gradient_checkpointing_enable()
            if hasattr(self.backbone, "enable_input_require_grads"):
                self.backbone.enable_input_require_grads()

        base_hidden = int(self.backbone.config.hidden_size)
        if pooling == "attn":
            self.pool = AttnPool(base_hidden, dropout=dropout)
        else:
            self.pool = None

        self.head = nn.Sequential(
            nn.Linear(base_hidden, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, num_targets)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state

        if self.pooling == "attn":
            pooled = self.pool(x, attention_mask=attention_mask)
        elif self.pooling == "last_token":
            batch_idx = torch.arange(input_ids.shape[0], device=input_ids.device)
            last_token_idx = attention_mask.sum(1).long() - 1
            pooled = x[batch_idx, last_token_idx, :]
        else:
            pooled = _mean_pool(x, attention_mask=attention_mask)

        return self.head(pooled)


def focal_binary_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if gamma <= 0:
        return bce.mean()
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    focal_weight = (1.0 - p_t) ** gamma
    return (focal_weight * bce).mean()


def train_trial(train_dataset, val_loader, tokenizer, device, config, model_name):
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=4,
        pin_memory=(device.type == "cuda")
    )

    model = BelkaClassifier(
        model_name=model_name,
        pooling=config["pooling"],
        mlp_hidden_size=config["mlp_hidden_size"],
        dropout=config.get("dropout", 0.25),
        unfreeze_last_n=config["n_unfreeze"],
        tokenizer_len=len(tokenizer),
        num_targets=3
    ).to(device)

    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("head")]

    param_groups = []
    if len(backbone_params) > 0:
        param_groups.append({"params": backbone_params, "lr": config["backbone_lr"], "weight_decay": config["weight_decay"]})
    if len(head_params) > 0:
        param_groups.append({"params": head_params, "lr": config["head_lr"], "weight_decay": config["weight_decay"]})

    optimizer = torch.optim.AdamW(param_groups)
    epochs = config.get("epochs", 3)
    steps_per_epoch = 1500  # Streamed step budget
    total_steps = epochs * steps_per_epoch
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    metric_map = MultilabelAveragePrecision(num_labels=3, average="macro").to(device)

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            train_dataset.set_epoch(epoch)
            running_loss = []
            step = 0

            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids, attention_mask)
                loss = focal_binary_cross_entropy(logits, labels, gamma=config.get("focal_gamma", 2.0))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                running_loss.append(loss.item())

                step += 1
                if step >= steps_per_epoch:
                    break

            # Evaluate
            model.eval()
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    logits = model(input_ids, attention_mask)
                    val_preds.append(torch.sigmoid(logits))
                    val_targets.append(labels.long())

            preds_cat = torch.cat(val_preds, dim=0)
            targets_cat = torch.cat(val_targets, dim=0)
            val_map = float(metric_map(preds_cat, targets_cat).item())
            val_loss = float(focal_binary_cross_entropy(preds_cat, targets_cat.float(), gamma=0.0).item())

            wandb.log({
                "epoch": epoch,
                "train_loss": float(np.mean(running_loss)),
                "val_loss": val_loss,
                "val_map": val_map,
                "lr/head": scheduler.get_last_lr()[-1]
            })
    finally:
        del model
        del optimizer
        del scheduler
        del train_loader
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def main():
    parser = argparse.ArgumentParser(description="Belka Multi-target Bayesian / Grid Search Sweep")
    parser.add_argument("--model", type=str, default="170m", help="Model scale (170m, 380m, 1b, 3b) or Hugging Face repository ID")
    default_data_dir = "/mnt/weka/asafrastyan/belka/data/processed_full" if os.path.isdir("/mnt/weka/asafrastyan/belka/data/processed_full") else "./data/belka/processed_full"
    parser.add_argument("--data_dir", type=str, default=default_data_dir, help="Directory containing processed Belka parquet files")
    parser.add_argument("--project", type=str, default="chemlactica-belka-thesis", help="WandB project name")
    parser.add_argument("--entity", type=str, default=None, help="WandB entity name")
    parser.add_argument("--count", type=int, default=50, help="Number of sweep trials to execute")
    parser.add_argument("--sweep_id", type=str, default=None, help="Existing WandB sweep ID to resume")
    args = parser.parse_args()

    model_key = args.model.lower()
    checkpoint_path = MODEL_MAP.get(model_key, args.model)
    unfreeze_vals = UNFREEZE_CANDIDATES.get(model_key, [4, 8, 16])

    print(f"Initializing Belka Sweep for Model: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["[START_SMILES]", "[END_SMILES]"]})

    # Prepare datasets
    train_dataset = BelkaTrainDataset(tokenizer, data_dir=args.data_dir, max_len=256, seed=42)
    val_dataset = BelkaValDataset(tokenizer, data_dir=args.data_dir, max_len=256)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))

    sweep_config = {
        "method": "bayes",
        "name": f"belka-{model_key}-sweep",
        "metric": {"goal": "minimize", "name": "val_loss"},
        "parameters": {
            "head_lr": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},
            "backbone_lr": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3},
            "weight_decay": {"values": [0.001, 0.05, 0.1, 0.2]},
            "mlp_hidden_size": {"values": [512, 1024, 2048]},
            "pooling": {"values": ["last_token", "attn"]},
            "n_unfreeze": {"values": unfreeze_vals},
            "batch_size": {"values": [32, 64]},
            "focal_gamma": {"values": [0, 2]}
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
                "mlp_hidden_size": wandb.config.mlp_hidden_size,
                "pooling": wandb.config.pooling,
                "n_unfreeze": wandb.config.n_unfreeze,
                "batch_size": wandb.config.batch_size,
                "focal_gamma": wandb.config.focal_gamma,
                "dropout": 0.25,
                "epochs": 3
            }

            try:
                train_trial(train_dataset, val_loader, tokenizer, device, config, checkpoint_path)
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
