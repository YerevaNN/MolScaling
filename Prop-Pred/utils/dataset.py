import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SmilesDataset(Dataset):
    """
    Standard PyTorch dataset for SMILES string inputs with ChemLlama tokenization.
    Supports both single-target and multi-target tasks, with optional sample weights.
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        max_len: int = 128,
        smiles_col: str = "smiles",
        target_col: str = None,
        tasks: list[str] = None
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.tasks = tasks
        
        if "sample_weight" in self.dataframe.columns:
            self.weights = self.dataframe["sample_weight"].tolist()
        else:
            self.weights = [1.0] * len(self.dataframe)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, i: int):
        row = self.dataframe.iloc[i]
        smiles = str(row[self.smiles_col])
        formatted_smiles = f"[START_SMILES] {smiles} [END_SMILES]"
        encodings = self.tokenizer(
            formatted_smiles,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        weight = torch.tensor(self.weights[i], dtype=torch.float32)

        # Multi-task mode (e.g. Polaris 6 ADME targets)
        if self.tasks is not None:
            targets_list = []
            masks_list = []
            for t in self.tasks:
                val = row.get(t, np.nan)
                if pd.isna(val) or val == "" or val is None:
                    targets_list.append(0.0)
                    masks_list.append(0.0)
                else:
                    targets_list.append(float(val))
                    masks_list.append(1.0)
            
            return {
                "input_ids": encodings["input_ids"].squeeze(0),
                "attention_mask": encodings["attention_mask"].squeeze(0),
                "targets": torch.tensor(targets_list, dtype=torch.float32),
                "masks": torch.tensor(masks_list, dtype=torch.float32),
                "weights": weight
            }

        # Single-target mode (e.g. PXR pEC50)
        target_name = self.target_col if self.target_col is not None else "pEC50"
        val = row.get(target_name, np.nan)
        if pd.isna(val) or val == "" or val is None:
            target = torch.tensor(0.0, dtype=torch.float32)
            mask = torch.tensor(0.0, dtype=torch.float32)
        else:
            target = torch.tensor(float(val), dtype=torch.float32)
            mask = torch.tensor(1.0, dtype=torch.float32)

        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "target": target,
            "mask": mask,
            "weight": weight
        }


def collate_fn(batch):
    """
    Collate function supporting both single-target and multi-target batches.
    """
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)

    if "targets" in batch[0]:
        targets = torch.stack([b["targets"] for b in batch], dim=0)
        masks = torch.stack([b["masks"] for b in batch], dim=0)
        weights = torch.stack([b["weights"] for b in batch], dim=0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": targets,
            "masks": masks,
            "weights": weights
        }
    else:
        targets = torch.stack([b["target"] for b in batch], dim=0)
        masks = torch.stack([b["mask"] for b in batch], dim=0)
        weights = torch.stack([b["weight"] for b in batch], dim=0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": targets,
            "masks": masks,
            "weights": weights
        }
