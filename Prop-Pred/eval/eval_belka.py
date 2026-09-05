"""
Inference and Kaggle Competition Submission Generation for Leash-BELKA Benchmark.
Benchmark URL: https://www.kaggle.com/competitions/leash-BELKA/overview
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from rdkit import Chem

# Allow importing from sibling utils module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.models import _freeze_all_but_last_n, AttnPool, _mean_pool

MODEL_MAP = {
    "170m": "yerevann/chemllama-170m",
    "380m": "yerevann/ChemLlama-380M",
    "1b": "yerevann/ChemLlama-1B",
    "1.3b": "yerevann/ChemLlama-1B",
    "3b": "yerevann/ChemLlama-3B"
}

ARCH_DEFAULTS = {
    "170m": {"pooling": "attn", "mlp_hidden_size": 512, "n_unfreeze": 4},
    "380m": {"pooling": "last_token", "mlp_hidden_size": 1024, "n_unfreeze": 4},
    "1b": {"pooling": "last_token", "mlp_hidden_size": 1024, "n_unfreeze": 16},
    "3b": {"pooling": "attn", "mlp_hidden_size": 512, "n_unfreeze": 8}
}


class BelkaInferenceClassifier(nn.Module):
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


class BelkaTestDataset(Dataset):
    def __init__(self, smiles_list: list[str], tokenizer, max_len: int = 256):
        self.smiles = smiles_list
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx: int):
        prompt = f"[START_SMILES]{self.smiles[idx]}[END_SMILES]"
        e = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": e["input_ids"].squeeze(0),
            "attention_mask": e["attention_mask"].squeeze(0)
        }


def main():
    parser = argparse.ArgumentParser(description="Generate Kaggle Leash-BELKA Test Predictions")
    parser.add_argument("--model", type=str, default="380m", help="Model scale (170m, 380m, 1b, 3b)")
    parser.add_argument(
        "--test_parquet",
        type=str,
        default=os.environ.get("BELKA_TEST_PARQUET", "./data/belka/test.parquet"),
        help="Path to Kaggle test.parquet (or set via BELKA_TEST_PARQUET)"
    )
    parser.add_argument("--weights_dir", type=str, default=None, help="Directory with seed checkpoints")
    parser.add_argument("--seeds", nargs="+", type=int, default=[16, 42, 85], help="Seeds to evaluate/ensemble")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for FP16 inference")
    parser.add_argument("--output_csv", type=str, default=None, help="Path for Kaggle submission CSV")
    args = parser.parse_args()

    model_key = args.model.lower()
    model_name = MODEL_MAP.get(model_key, args.model)
    arch = ARCH_DEFAULTS.get(model_key, {"pooling": "last_token", "mlp_hidden_size": 512, "n_unfreeze": 4})

    weights_dir = args.weights_dir if args.weights_dir else f"./weights/belka/{model_key}"
    output_csv = args.output_csv if args.output_csv else f"./results/belka/submission_{model_key}.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["[START_SMILES]", "[END_SMILES]"]})

    print(f"Reading Kaggle test parquet from {args.test_parquet}...")
    test_df = pd.read_parquet(args.test_parquet, columns=["id", "molecule_smiles", "protein_name"])
    unique_smiles = test_df["molecule_smiles"].unique().tolist()
    print(f"Unique test molecules: {len(unique_smiles)} (Total rows: {len(test_df)})")

    # Canonicalize smiles for inference
    canon_smiles = []
    for s in tqdm(unique_smiles, desc="Canonicalizing test SMILES"):
        try:
            canon_smiles.append(Chem.CanonSmiles(s.replace("[Dy]", "[H]")))
        except Exception:
            canon_smiles.append(s)

    test_ds = BelkaTestDataset(canon_smiles, tokenizer, max_len=256)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_seed_preds = []
    for seed in args.seeds:
        weight_path = os.path.join(weights_dir, f"belka_{model_key}_seed_{seed}.pt")
        if not os.path.exists(weight_path):
            alt_path = os.path.join(weights_dir, f"best_llama_380_seed_{seed}_max_grid.pt")
            if os.path.exists(alt_path):
                weight_path = alt_path
            else:
                print(f"Warning: Checkpoint for seed {seed} not found at {weight_path}. Skipping.")
                continue

        print(f"\nRunning FP16 batched inference for Seed {seed} from {weight_path}...")
        model = BelkaInferenceClassifier(
            model_name=model_name,
            pooling=arch["pooling"],
            mlp_hidden_size=arch["mlp_hidden_size"],
            dropout=0.25,
            unfreeze_last_n=arch["n_unfreeze"],
            tokenizer_len=len(tokenizer),
            num_targets=3
        ).to(device)

        sd = torch.load(weight_path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        keys_to_remove = [k for k in sd.keys() if "loss" in k or "tr_" in k or "val_" in k]
        for k in keys_to_remove:
            del sd[k]
        model.load_state_dict(sd, strict=False)
        model.eval()
        model.half()

        seed_preds = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Seed {seed} Forward Pass"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids, attention_mask)
                probs = torch.sigmoid(logits).cpu().float().numpy()
                seed_preds.append(probs)

        seed_preds_cat = np.concatenate(seed_preds, axis=0)  # Shape: (num_unique, 3)
        all_seed_preds.append(seed_preds_cat)

        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(all_seed_preds) == 0:
        print("Error: No checkpoints were successfully evaluated.")
        return

    # Average predictions across seeds if multiple evaluated
    avg_preds = np.mean(all_seed_preds, axis=0)

    pred_df = pd.DataFrame({
        "molecule_smiles": unique_smiles,
        "BRD4": avg_preds[:, 0],
        "HSA": avg_preds[:, 1],
        "sEH": avg_preds[:, 2],
    })

    print("Merging predictions into Kaggle test format...")
    merged_df = test_df.merge(pred_df, on="molecule_smiles", how="left")
    merged_df["binds"] = 0.0

    brd4_mask = merged_df["protein_name"] == "BRD4"
    hsa_mask = merged_df["protein_name"] == "HSA"
    seh_mask = merged_df["protein_name"] == "sEH"

    merged_df.loc[brd4_mask, "binds"] = merged_df.loc[brd4_mask, "BRD4"]
    merged_df.loc[hsa_mask, "binds"] = merged_df.loc[hsa_mask, "HSA"]
    merged_df.loc[seh_mask, "binds"] = merged_df.loc[seh_mask, "sEH"]

    sub = merged_df[["id", "binds"]]
    sub.to_csv(output_csv, index=False)
    print(f"\nSuccessfully generated Kaggle submission CSV at: {output_csv}")
    print("\nTo submit to Kaggle CLI:")
    print(f"  kaggle competitions submit -c leash-BELKA -f {output_csv} -m \"ChemLlama-{model_key.upper()} multi-seed\"")


if __name__ == "__main__":
    main()
