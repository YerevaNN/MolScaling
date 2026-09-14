import os
import sys
import pandas as pd
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.models import ChemlacticaRegressor, ChemlacticaMultiTaskRegressor
from train.train_belka import BelkaClassifier, FocalLoss
from utils.curation import standardize_tautomers_and_smiles, exclude_reactive_electrophiles, compute_sample_weights
from utils.metrics import masked_mae_loss, masked_mse_loss

def test_pxr_pipeline():
    print("\n--- Testing PXR Pipeline ---")
    df = pd.DataFrame({
        "SMILES": ["CC(=O)OC1=CC=CC=C1C(=O)O", "CCO", "C1=CC=CC=C1", "[NX3]-[CX3](=[OX1])-[CX3]=[CX3]"],
        "pEC50": [5.5, 4.2, 6.1, 7.0]
    })
    print("1. Data Curation...")
    df = standardize_tautomers_and_smiles(df, smiles_col_in="SMILES")
    df = exclude_reactive_electrophiles(df, smiles_col="smiles")
    df = compute_sample_weights(df, smiles_col="smiles")
    assert len(df) == 3, f"Expected 3 valid molecules, got {len(df)}"
    print("Curation passed.")

    print("2. Model Init (ChemlacticaRegressor 170m)...")
    tokenizer = AutoTokenizer.from_pretrained("yerevann/chemllama-170m")
    tokenizer.add_special_tokens({"additional_special_tokens": ["[START_SMILES]", "[END_SMILES]"]})
    
    model = ChemlacticaRegressor(
        model_name="yerevann/chemllama-170m",
        pooling="last_token",
        mlp_hidden_size=256,
        mlp_layers=1,
        dropout=0.1,
        unfreeze_last_n=2,
        tokenizer_len=len(tokenizer),
        use_gradient_checkpointing=False
    )
    
    print("3. Forward & Backward Pass...")
    input_ids = torch.randint(0, len(tokenizer), (2, 32))
    attention_mask = torch.ones((2, 32))
    targets = torch.tensor([5.5, 4.2])
    masks = torch.tensor([1.0, 1.0])
    weights = torch.tensor([1.0, 0.4])
    
    preds = model(input_ids, attention_mask)
    loss, _ = masked_mae_loss(preds, targets, masks, weights=weights)
    loss.backward()
    print("PXR Pipeline Passed!")


def test_polaris_pipeline():
    print("\n--- Testing Polaris Pipeline ---")
    print("1. Model Init (ChemlacticaMultiTaskRegressor 170m)...")
    tokenizer = AutoTokenizer.from_pretrained("yerevann/chemllama-170m")
    tokenizer.add_special_tokens({"additional_special_tokens": ["[START_SMILES]", "[END_SMILES]"]})
    
    tasks = ["HLM", "Sol", "MDR1", "RLM", "hPPB", "rPPB"]
    model = ChemlacticaMultiTaskRegressor(
        model_name="yerevann/chemllama-170m",
        tasks=tasks,
        pooling="attn",
        mlp_hidden_size=256,
        mlp_layers=1,
        dropout=0.1,
        unfreeze_last_n=2,
        tokenizer_len=len(tokenizer),
        use_gradient_checkpointing=False
    )
    
    print("2. Forward & Backward Pass (Multitask)...")
    input_ids = torch.randint(0, len(tokenizer), (2, 32))
    attention_mask = torch.ones((2, 32))
    
    targets = torch.randn(2, 6)
    masks = torch.ones(2, 6)
    
    preds = model(input_ids, attention_mask)
    loss, _ = masked_mse_loss(preds, targets, masks, tasks=tasks)
    loss.backward()
    print("Polaris Pipeline Passed!")


def test_belka_pipeline():
    print("\n--- Testing Belka Pipeline ---")
    print("1. Model Init (BelkaClassifier 170m)...")
    tokenizer = AutoTokenizer.from_pretrained("yerevann/chemllama-170m")
    tokenizer.add_special_tokens({"additional_special_tokens": ["[START_SMILES]", "[END_SMILES]"]})
    
    model = BelkaClassifier(
        model_name="yerevann/chemllama-170m",
        pooling="mean",
        mlp_hidden_size=256,
        dropout=0.1,
        unfreeze_last_n=2,
        tokenizer_len=len(tokenizer),
        num_targets=3
    )
    
    print("2. Focal Loss & Backward Pass...")
    input_ids = torch.randint(0, len(tokenizer), (2, 32))
    attention_mask = torch.ones((2, 32))
    labels = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    
    loss_fn = FocalLoss(gamma=2.0)
    logits = model(input_ids, attention_mask)
    loss = loss_fn(logits, labels)
    loss.backward()
    print("Belka Pipeline Passed!")


if __name__ == "__main__":
    print("Starting Prop-Pred Sanity Checks...\n")
    test_pxr_pipeline()
    test_polaris_pipeline()
    test_belka_pipeline()
    print("\nAll Sanity Checks completed successfully!")
