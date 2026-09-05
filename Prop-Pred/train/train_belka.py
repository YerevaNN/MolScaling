"""
Multi-Seed Training for Belka Multi-target Classification across ChemLlama scales.
"""
import os
import sys
import glob
import random
import hashlib
import argparse
import yaml
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torchmetrics.classification import MultilabelAveragePrecision
from tqdm import tqdm

# Allow importing from sibling utils module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.models import _freeze_all_but_last_n, AttnPool, _mean_pool
from utils.metrics import _set_seed


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


def train_single_seed(tokenizer, device, config, seed, data_dir, weight_path):
    print(f"\n==========================================")
    print(f"Starting Belka Training for Seed {seed}")
    print(f"==========================================")
    _set_seed(seed)

    train_dataset = BelkaTrainDataset(tokenizer, data_dir=data_dir, max_len=256, seed=seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=4,
        pin_memory=(device.type == "cuda")
    )

    model = BelkaClassifier(
        model_name=config["model_name"],
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
    steps_per_epoch = 1500
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(config.get("warmup_ratio", 0.1) * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            train_dataset.set_epoch(epoch)
            running_loss = []
            step = 0

            pbar = tqdm(train_loader, total=steps_per_epoch, desc=f"Seed {seed} Epoch {epoch}/{epochs}", leave=False)
            for batch in pbar:
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

            print(f"[Seed {seed} Epoch {epoch}] Train Loss: {np.mean(running_loss):.4f}")

        os.makedirs(os.path.dirname(weight_path), exist_ok=True)
        torch.save({"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}, weight_path)
        print(f"Saved Belka checkpoint for Seed {seed}: {weight_path}")
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
    parser = argparse.ArgumentParser(description="Multi-Seed Training for Belka Multi-target Classification")
    parser.add_argument("--model", type=str, default="380m", help="Model scale (170m, 380m, 1b, 3b)")
    parser.add_argument("--config", type=str, default=None, help="Path to custom config YAML")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("BELKA_DATA_DIR", "./data/belka/processed_full"),
        help="Directory with processed parquet files (or set via BELKA_DATA_DIR)"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[16, 42, 85], help="List of random seeds")
    parser.add_argument("--output_dir", type=str, default="./weights/belka", help="Directory to save model checkpoints")
    args = parser.parse_args()

    model_key = args.model.lower()
    config_path = args.config if args.config else os.path.join(os.path.dirname(__file__), "configs", "belka.yaml")

    with open(config_path, "r") as f:
        all_configs = yaml.safe_load(f)

    if model_key not in all_configs:
        raise ValueError(f"Model key '{model_key}' not found in {config_path}. Available: {list(all_configs.keys())}")

    config = all_configs[model_key]
    model_name = config["model_name"]
    print(f"Configuring Belka Multi-Seed Training for {model_key.upper()} ({model_name})")
    print(f"Seeds: {args.seeds}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["[START_SMILES]", "[END_SMILES]"]})

    save_dir = os.path.join(args.output_dir, model_key)
    for seed in args.seeds:
        weight_path = os.path.join(save_dir, f"belka_{model_key}_seed_{seed}.pt")
        train_single_seed(tokenizer, device, config, seed, args.data_dir, weight_path)

    print("\nAll Belka seeds completed successfully.")


if __name__ == "__main__":
    main()
