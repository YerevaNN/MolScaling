"""
Multi-Seed Evaluation, Local Metric Calculation, and Optional Polaris Hub Submission for Polaris ADME Benchmark.
"""
import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import polaris as po

# Allow importing from sibling utils module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.models import ChemlacticaMultiTaskRegressor

MODEL_MAP = {
    "170m": "yerevann/ChemLlama-170M",
    "380m": "yerevann/ChemLlama-380M",
    "1b": "yerevann/ChemLlama-1B",
    "1.3b": "yerevann/ChemLlama-1B",
    "3b": "yerevann/ChemLlama-3B"
}

ARCH_DEFAULTS = {
    "170m": {"pooling": "last_token", "mlp_hidden_size": 512, "n_unfreeze": 8, "sub_prefix": "chemllama170"},
    "380m": {"pooling": "last_token", "mlp_hidden_size": 512, "n_unfreeze": 16, "sub_prefix": "Chemllama380"},
    "1b": {"pooling": "last_token", "mlp_hidden_size": 512, "n_unfreeze": 16, "sub_prefix": "Chemllama13"},
    "3b": {"pooling": "last_token", "mlp_hidden_size": 512, "n_unfreeze": 24, "sub_prefix": "Chemllama_3B"}
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate Polaris ADME Models on Test Set")
    parser.add_argument("--model", type=str, default="3b", help="Model scale (170m, 380m, 1b, 3b)")
    parser.add_argument("--benchmark", type=str, default="polaris/adme-fang-r-1", help="Polaris benchmark identifier")
    parser.add_argument("--weights_dir", type=str, default=None, help="Directory containing seed checkpoints")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 81, 75, 95, 114], help="Seeds to evaluate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for test inference")
    parser.add_argument("--output_dir", type=str, default="./results/polaris", help="Directory to save metric tables and predictions")
    parser.add_argument("--upload_to_hub", action="store_true", help="Upload predictions to Polaris Hub (requires polaris login)")
    parser.add_argument("--owner", type=str, default=None, help="Polaris Hub organization or username for upload")
    args = parser.parse_args()

    model_key = args.model.lower()
    model_name = MODEL_MAP.get(model_key, args.model)
    arch = ARCH_DEFAULTS.get(model_key, {"pooling": "last_token", "mlp_hidden_size": 512, "n_unfreeze": 16, "sub_prefix": f"chemllama_{model_key}"})

    weights_dir = args.weights_dir if args.weights_dir else f"./weights/polaris/{model_key}"
    os.makedirs(args.output_dir, exist_ok=True)
    preds_dir = os.path.join(args.output_dir, "predictions")
    os.makedirs(preds_dir, exist_ok=True)

    print(f"Loading Polaris Benchmark: {args.benchmark}...")
    benchmark = po.load_benchmark(args.benchmark)
    _, test_split = benchmark.get_train_test_split()
    test_smiles = test_split.inputs
    num_samples = len(test_smiles)
    print(f"Test split contains {num_samples} molecules.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["[START_SMILES]", "[END_SMILES]"]})

    all_seed_results = []
    for seed in args.seeds:
        weight_path = os.path.join(weights_dir, f"polaris_{model_key}_seed_{seed}_best.pt")
        if not os.path.exists(weight_path):
            alt_path = os.path.join(weights_dir, f"polaris_max_grid_3_seed_{seed}_best.pt")
            if os.path.exists(alt_path):
                weight_path = alt_path
            else:
                print(f"Warning: Checkpoint for seed {seed} not found at {weight_path}. Skipping.")
                continue

        print(f"\nEvaluating Seed {seed} from {weight_path}...")
        model = ChemlacticaMultiTaskRegressor(
            model_name=model_name,
            tasks=INTERNAL_TASKS,
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
        model.eval()

        seed_preds = {label: [] for label in benchmark.target_cols}
        with torch.no_grad():
            for i in tqdm(range(0, num_samples, args.batch_size), desc=f"Seed {seed} Inference"):
                batch_smiles = test_smiles[i:i + args.batch_size]
                formatted_texts = [f"[START_SMILES] {smi} [END_SMILES]" for smi in batch_smiles]
                encodings = tokenizer(
                    formatted_texts, truncation=True, padding="max_length", max_length=128, return_tensors="pt"
                )
                input_ids = encodings["input_ids"].to(device)
                attention_mask = encodings["attention_mask"].to(device)

                with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                    preds_dict = model(input_ids, attention_mask)
                    for polaris_label in benchmark.target_cols:
                        internal_name = POLARIS_TO_INTERNAL.get(polaris_label)
                        seed_preds[polaris_label].extend(preds_dict[internal_name].detach().cpu().numpy().tolist())

        # Save local predictions
        pred_save_path = os.path.join(preds_dir, f"polaris_{model_key}_seed_{seed}.json")
        with open(pred_save_path, "w") as f:
            json.dump(seed_preds, f)
        print(f"Saved predictions to {pred_save_path}")

        # Local benchmark evaluation
        try:
            benchmark_results = benchmark.evaluate(seed_preds)
            sub_name = f"{arch['sub_prefix']}_{seed}"
            benchmark_results.name = sub_name
            print(f"Computed local benchmark evaluation for {sub_name}.")

            if args.upload_to_hub:
                print(f"Uploading {sub_name} to Polaris Hub (owner: {args.owner})...")
                benchmark_results.upload_to_hub(owner=args.owner)
                print("Polaris Hub upload successful.")
        except Exception as e:
            print(f"Note: Local benchmark metric calculation exception: {e}")

        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nPolaris evaluation completed for all available seeds.")


if __name__ == "__main__":
    main()
