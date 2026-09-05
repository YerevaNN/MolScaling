#!/usr/bin/env python3
"""
Preprocess and partition raw Kaggle Leash-BELKA train.parquet into
streaming multi-target parquets with 3% scaffold validation split.
"""

import argparse
import os
import numpy as np
import pandas as pd
import pyarrow as pa
import dask.dataframe as dd
from rdkit import Chem


def prepare_belka_data(raw_parquet: str, out_dir: str, seed: int = 42, npartitions: int = 16):
    os.makedirs(out_dir, exist_ok=True)
    sentinel = os.path.join(out_dir, ".ready")
    if os.path.exists(sentinel):
        print(f"Data already prepared at {out_dir} (found .ready sentinel).")
        return

    if not os.path.exists(raw_parquet):
        raise FileNotFoundError(f"Raw parquet file not found at {raw_parquet}")

    print(f"Reading raw parquet from {raw_parquet} ...")
    df = pd.read_parquet(raw_parquet)

    print("Pivoting multi-target binding labels (BRD4, HSA, sEH) ...")
    df = df.rename(columns={
        "buildingblock1_smiles": "block1",
        "buildingblock2_smiles": "block2",
        "buildingblock3_smiles": "block3",
        "molecule_smiles": "smiles",
    })
    df = df.pivot(
        index=["block1", "block2", "block3", "smiles"],
        columns="protein_name",
        values="binds"
    ).reset_index().fillna(0)

    for col in ["BRD4", "HSA", "sEH"]:
        df[col] = df[col].astype(np.int8)

    print("Generating 3% scaffold validation split based on building blocks ...")
    rng = np.random.default_rng(seed)
    blocks = list(set(df.block1) | set(df.block2) | set(df.block3))
    rng.shuffle(blocks)
    val_blocks = set(blocks[: int(0.03 * len(blocks))])

    df["subset"] = df.apply(
        lambda r: int(len({r.block1, r.block2, r.block3} & val_blocks) > 0),
        axis=1
    ).astype(np.int8)
    df = df.drop(columns=["block1", "block2", "block3"])

    print("Canonicalizing SMILES (replacing Dy attachment points with H) ...")
    df["smiles"] = df["smiles"].apply(
        lambda s: Chem.CanonSmiles(s.replace("[Dy]", "[H]"))
    )
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"Saving partitioned parquet dataset to {out_dir} with {npartitions} partitions ...")
    dd.from_pandas(df, npartitions=npartitions).to_parquet(
        out_dir,
        schema={
            "smiles": pa.string(),
            "BRD4": pa.int8(),
            "HSA": pa.int8(),
            "sEH": pa.int8(),
            "subset": pa.int8()
        },
    )
    open(sentinel, "w").close()
    print("Pre-processing complete. Sentinel file (.ready) created.")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Kaggle Leash-BELKA raw parquet into partitioned format")
    parser.add_argument("--raw_parquet", type=str, default="./data/belka/train.parquet", help="Path to raw Kaggle train.parquet")
    parser.add_argument("--out_dir", type=str, default="./data/belka/processed_full", help="Output directory for partitioned parquets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for scaffold split")
    parser.add_argument("--npartitions", type=int, default=16, help="Number of parquet partitions")
    args = parser.parse_args()

    prepare_belka_data(args.raw_parquet, args.out_dir, seed=args.seed, npartitions=args.npartitions)


if __name__ == "__main__":
    main()
