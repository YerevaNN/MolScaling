import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Cluster import Butina
from rdkit import DataStructs


def _ensure_smiles_column(df: pd.DataFrame, benchmark_obj=None) -> tuple[pd.DataFrame, str]:
    """
    Finds and standardizes the SMILES column name to 'smiles'.
    """
    if "smiles" in df.columns:
        return df, "smiles"
    if benchmark_obj is not None:
        input_cols = list(getattr(benchmark_obj, "input_cols", []))
        if len(input_cols) >= 1 and input_cols[0] in df.columns:
            df = df.rename(columns={input_cols[0]: "smiles"})
            return df, "smiles"
    for candidate in ["canonical_smiles", "smiles_canonical", "SMILES", "Smiles"]:
        if candidate in df.columns:
            df = df.rename(columns={candidate: "smiles"})
            return df, "smiles"
    raise ValueError(f"Could not find SMILES column. Available columns: {list(df.columns)}")


def standardize_tautomers_and_smiles(df: pd.DataFrame, smiles_col_in: str = "SMILES") -> pd.DataFrame:
    """
    Standardizes SMILES strings using RDKit's TautomerEnumerator 
    to convert enol forms to keto forms and canonicalize.
    """
    print("Standardizing SMILES and resolving tautomers...")
    enumerator = rdMolStandardize.TautomerEnumerator()
    standardized_smiles = []

    for s in tqdm(df[smiles_col_in], desc="Tautomer standardization"):
        mol = Chem.MolFromSmiles(str(s).split()[0])  # split to handle CXSMILES robustly
        if mol is not None:
            try:
                canon_mol = enumerator.Canonicalize(mol)
                std_s = Chem.MolToSmiles(canon_mol, canonical=True)
                standardized_smiles.append(std_s)
            except Exception:
                standardized_smiles.append(None)
        else:
            standardized_smiles.append(None)

    df = df.copy()
    df["smiles"] = standardized_smiles
    initial_len = len(df)
    df = df.dropna(subset=["smiles"]).reset_index(drop=True)
    final_len = len(df)
    print(f"Standardization complete: {initial_len - final_len} invalid/failed structures removed.")
    return df


def exclude_reactive_electrophiles(df: pd.DataFrame, smiles_col: str = "smiles") -> pd.DataFrame:
    """
    Filters out molecules containing highly reactive electrophiles:
    - Acrylamides: [NX3]-[CX3](=[OX1])-[CX3]=[CX3]
    - Acrylates: [OX2]-[CX3](=[OX1])-[CX3]=[CX3]
    - Aldehydes: [CX3H1](=[OX1])[#6,#1]
    """
    print("Filtering out reactive electrophiles (acrylamides, acrylates, aldehydes)...")
    acrylamide = Chem.MolFromSmarts("[NX3]-[CX3](=[OX1])-[CX3]=[CX3]")
    acrylate = Chem.MolFromSmarts("[OX2]-[CX3](=[OX1])-[CX3]=[CX3]")
    aldehyde = Chem.MolFromSmarts("[CX3H1](=[OX1])[#6,#1]")
    patterns = [acrylamide, acrylate, aldehyde]

    keep_mask = []
    for s in tqdm(df[smiles_col], desc="Reactive electrophile filtering"):
        mol = Chem.MolFromSmiles(str(s))
        if mol is not None:
            has_match = any(mol.HasSubstructMatch(pat) for pat in patterns)
            keep_mask.append(not has_match)
        else:
            keep_mask.append(False)

    df = df[keep_mask].reset_index(drop=True)
    print(f"Filtering complete: {len(keep_mask) - sum(keep_mask)} reactive electrophile(s) removed. Remaining: {len(df)}")
    return df


def compute_sample_weights(df: pd.DataFrame, smiles_col: str = "smiles") -> pd.DataFrame:
    """
    Computes sample weights based on Molecular Weight and standard error,
    appending a 'sample_weight' column to the DataFrame.
    """
    print("Computing sample weights...")
    df = df.copy()
    weights = []
    for _, row in df.iterrows():
        w = 1.0
        s = str(row[smiles_col])
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            # 1. Extreme Molecular Weight Check (< 150 or > 600) -> reduced weight of 0.4
            mw = Descriptors.ExactMolWt(mol)
            if mw < 150.0 or mw > 600.0:
                w = min(w, 0.4)

            # 2. High measurement variance check (> 0.15) -> reduced weight of 0.4
            std_err = row.get("pEC50_std_error", 0.0)
            if pd.notna(std_err) and float(std_err) > 0.15:
                w = min(w, 0.4)
        weights.append(w)
    df["sample_weight"] = weights
    return df


def butina_clustering_split(df: pd.DataFrame, smiles_col: str = "smiles", distance_cutoff: float = 0.4, test_ratio: float = 0.10):
    """
    Performs Butina scaffold-based clustering split using ECFP4 fingerprints.
    """
    print(f"Generating ECFP4 fingerprints for Butina split (cutoff={distance_cutoff})...")
    fps = []
    valid_indices = []
    for idx, s in enumerate(df[smiles_col]):
        mol = Chem.MolFromSmiles(str(s))
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(fp)
            valid_indices.append(idx)

    df = df.iloc[valid_indices].reset_index(drop=True)
    nPts = len(fps)

    dists = []
    for i in range(1, nPts):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        for s in sims:
            dists.append(1.0 - s)

    clusters = Butina.ClusterData(dists, nPts, distance_cutoff, isDistData=True)
    sorted_clusters = sorted(clusters, key=lambda x: (len(x), x[0]), reverse=True)

    train_indices = []
    test_indices = []
    target_test = int(test_ratio * nPts)

    for cluster in sorted_clusters:
        if len(test_indices) + len(cluster) <= target_test:
            test_indices.extend(cluster)
        else:
            train_indices.extend(cluster)

    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    print(f"Butina split completed: Train={len(train_df)} molecules, Test/Val={len(test_df)} molecules.")
    return train_df, test_df


def create_stratified_folds(df: pd.DataFrame, target_cols: list[str], n_splits: int = 5) -> list[dict]:
    """
    Creates multi-task profile stratified folds for uniform cross-validation.
    """
    discrete_df = pd.DataFrame(index=df.index)
    for col in target_cols:
        median_val = df[col].dropna().median()

        def map_val(val, med=median_val):
            if pd.isna(val):
                return "Missing"
            elif val < med:
                return "Low"
            else:
                return "High"

        discrete_df[col] = df[col].apply(map_val)

    discrete_df["profile"] = discrete_df[target_cols].apply(lambda row: "_".join(row.values.astype(str)), axis=1)
    profile_counts = discrete_df["profile"].value_counts()
    discrete_df["profile_count"] = discrete_df["profile"].map(profile_counts)

    sorted_idx = discrete_df.sort_values("profile_count").index.tolist()
    folds = [[] for _ in range(n_splits)]
    counts = {i: {col: {"High": 0, "Low": 0, "Missing": 0} for col in target_cols} for i in range(n_splits)}

    for idx in sorted_idx:
        row_labels = discrete_df.loc[idx, target_cols]
        scores = []
        for i in range(n_splits):
            score = 0
            for col in target_cols:
                label = row_labels[col]
                score += counts[i][col][label]
            scores.append((score, len(folds[i]), i))
        scores.sort()
        best_fold = scores[0][2]
        folds[best_fold].append(idx)
        for col in target_cols:
            label = row_labels[col]
            counts[best_fold][col][label] += 1

    fold_dfs = []
    for i in range(n_splits):
        val_idx = folds[i]
        train_idx = [idx for j in range(n_splits) if j != i for idx in folds[j]]
        fold_dfs.append({
            "train_df": df.loc[train_idx].copy().reset_index(drop=True),
            "val_df": df.loc[val_idx].copy().reset_index(drop=True)
        })
    return fold_dfs


def get_train_test_dfs(benchmark, rename_targets=None, test_same_shape_as_train=False):
    """
    Helper to extract pandas DataFrames from a Polaris benchmark object.
    """
    train_subset, test_subset = benchmark.get_train_test_split()
    train_df = train_subset.as_dataframe()
    if rename_targets is not None:
        rename_map = {k: v for k, v in rename_targets.items() if k in train_df.columns}
        train_df = train_df.rename(columns=rename_map)
    if not test_same_shape_as_train:
        test_df = test_subset.as_dataframe()
        if rename_targets is not None:
            rename_map = {k: v for k, v in rename_targets.items() if k in test_df.columns}
            test_df = test_df.rename(columns=rename_map)
        return train_df, test_df

    target_cols = list(benchmark.target_cols)
    input_cols = list(benchmark.input_cols)
    test_inputs = test_subset.inputs
    if isinstance(test_inputs, dict):
        test_df = pd.DataFrame(test_inputs)
    else:
        inp_name = input_cols[0] if input_cols else "input"
        test_df = pd.DataFrame({inp_name: test_inputs})
    for col in target_cols:
        test_df[col] = np.nan
    if rename_targets is not None:
        rename_map = {k: v for k, v in rename_targets.items() if k in test_df.columns}
        test_df = test_df.rename(columns=rename_map)
    test_df = test_df.reindex(columns=train_df.columns)
    return train_df, test_df
