# Property Prediction (`Prop-Pred`) Track

This track evaluates language model scaling for molecular property prediction across four model scales: **ChemLlama-170M**, **ChemLlama-380M**, **ChemLlama-1B**, and **ChemLlama-3B** on three benchmark suites:
* **Polaris ADME** (6-target multi-task ADME benchmark: HLM, Solubility, MDR1-MDCK, RLM, hPPB, rPPB)
* **PXR Regression** (Pregnane X Receptor $p\text{EC}_{50}$ regression with RDKit tautomer standardization & Butina split)
* **Leash-BELKA** (Multi-target small molecule-protein binding classification for BRD4, HSA, and sEH)

---

## Datasets

1. **Polaris ADME (`polaris/adme-fang-r-1`):**
   * **Automatic Download:** Managed via the Polaris SDK. Running `po.load_benchmark("polaris/adme-fang-r-1")` automatically downloads and caches the data locally.

2. **PXR (`Prop-Pred/data/pxr/`):**
   * Pre-curated training and test sets are included directly in the repository:
     * `Prop-Pred/data/pxr/pxr_train.csv`
     * `Prop-Pred/data/pxr/pxr_test.csv`

3. **Leash-BELKA:**
   * Download the official competition parquet files from Kaggle:
     ```bash
     kaggle competitions download -c leash-BELKA -p ./data/belka/
     unzip ./data/belka/leash-BELKA.zip -d ./data/belka/
     ```

---

## Quickstart Workflows

### 1. Hyperparameter Sweeps (`greed-search/`)
Bayesian sweeps over learning rates, unfreezing depths, pooling strategies, batch sizes, and weight decays:

```bash
# PXR sweep
python Prop-Pred/greed-search/search_pxr.py --model 380m --count 100

# Polaris ADME sweep (stratified fold 0)
python Prop-Pred/greed-search/search_polaris.py --model 3b --count 100

# Belka sweep
python Prop-Pred/greed-search/search_belka.py --model 1b --count 50
```

---

### 2. Multi-Seed Training (`train/`)
Trains on 100% of the training dataset across random seeds using the optimal hyperparameters stored in `train/configs/`:

```bash
# Train PXR across 5 random seeds (42, 81, 75, 95, 114)
python Prop-Pred/train/train_pxr.py --model 380m

# Train Polaris across 5 random seeds
python Prop-Pred/train/train_polaris.py --model 3b

# Train Belka across seeds (16, 42, 85)
python Prop-Pred/train/train_belka.py --model 1b
```

---

### 3. Evaluation & Benchmark Submissions (`eval/`)

#### A. PXR Test Evaluation
Evaluates checkpoints on the unblinded test set and exports statistical summaries ($\text{Mean}$, $\text{Std}$, and Student's $t$ 95% Confidence Interval):
```bash
python Prop-Pred/eval/eval_pxr.py --model 380m
```
*Outputs:* `./results/pxr_380m_variance_metrics.csv`

#### B. Polaris ADME Evaluation & Hub Upload
Computes benchmark metrics locally and saves individual prediction JSONs:
```bash
# Local evaluation (reproducible offline without credentials)
python Prop-Pred/eval/eval_polaris.py --model 3b

# Optional: Upload to Polaris Hub profile (requires 'polaris login')
python Prop-Pred/eval/eval_polaris.py --model 3b --upload_to_hub --owner <your-username>
```
*Benchmark URL:* [https://polarishub.io/benchmarks/polaris/adme-fang-r-1](https://polarishub.io/benchmarks/polaris/adme-fang-r-1)

#### C. Belka Kaggle Submission
Runs fast batched FP16 inference on `test.parquet` and exports the official Kaggle submission CSV:
```bash
python Prop-Pred/eval/eval_belka.py --model 380m
```
*Submit to Kaggle via CLI:*
```bash
kaggle competitions submit -c leash-BELKA -f ./results/belka/submission_380m.csv -m "ChemLlama-380M multi-seed"
```
