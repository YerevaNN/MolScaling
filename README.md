# MolScaling: Scaling Language Models for Molecular Design & Discovery

**MolScaling** is a research framework investigating the scaling laws and empirical performance of chemical language models across diverse molecular design and discovery tasks.

The framework explores model scaling across multiple scales (**ChemLlama 170M, 380M, 1B, and 3B**) spanning three primary research tracks:

```
MolScaling/
├── Prop-Pred/      # Molecular property prediction (Regression, ADME, Binding)
├── Cond-Gen/       # Conditional molecular generation and optimization
├── PMO-Dock/       # Molecular docking evaluation suite
└── requirements.txt
```

---

## 🔬 Research Tracks

### 1. Molecular Property Prediction (`Prop-Pred/`)
Fine-tuning and evaluating chemical language models for downstream property prediction across four model scales:
* **Polaris ADME:** Multi-task ADME benchmark (`polaris/adme-fang-r-1`) spanning 6 clinical and pharmacokinetics endpoints (HLM, Solubility, MDR1-MDCK, RLM, hPPB, rPPB).
* **PXR Regression:** Pregnane X Receptor $p\text{EC}_{50}$ agonist potency prediction with RDKit tautomer standardization and Butina clustering splits.
* **Leash-BELKA:** Ultra-large-scale small molecule-protein binding classification across three targets (BRD4, HSA, sEH).

Features Bayesian hyperparameter search, multi-seed full-dataset training, and rigorous statistical variance analysis.  
👉 **For detailed instructions and quickstart commands, see the [Prop-Pred README](Prop-Pred/README.md).**

### 2. Conditional Molecular Generation (`Cond-Gen/`)
Property-conditioned SMILES generation with ChemLlama. Generated molecules are scored with **PMO-Dock `benchmark`** computers (QED, SA, Tanimoto similarity) so Cond-Gen matches the docking-track oracles. SA is `benchmark.computers.property_computers.compute_sas` (Ertl/Landrum `sascorer` + `benchmark/synthesizability/fpscores.pkl.gz`).

```
Cond-Gen/
  config.yaml          # seed, decoding, device, Hub aliases, task → properties
  run.py               # generate + score + plots
  cond_gen.py          # scoring / generation / figures
  prompts/             # frozen evaluation prompts (JSON arrays)
```

`--model` and `--task(s)` stay on the CLI; everything else is in `config.yaml`. `--model all` / `--models all` runs every alias in `models:`. `--task all` / `--tasks all` (or omitting tasks) runs every key in `tasks:`.

```bash
python Cond-Gen/run.py --model chemllama-170m --task qed
python Cond-Gen/run.py --model all --tasks all
python Cond-Gen/run.py --models chemllama-170m chemllama-380m --tasks qed sas qed_sas
python Cond-Gen/run.py --plot
```

| CLI name | Prompt tags |
|----------|-------------|
| `qed` / `sas` / `similarity_random` | single |
| `qed_sas` | `[QED]x[/QED][SAS]y[/SAS][SMILES]` |
| `sas_qed` | same `(x,y)`, tags swapped |
| `qed_sas_similarity_random` | QED, SA, then SIMILAR |
| `sas_qed_similarity_random` | SA, QED, then SIMILAR |

QED and SA in joint prompts come from one ZINC250K molecule. Similarity target is Uniform`[0,1]` vs a ZINC reference SMILES.

Outputs: `Cond-Gen/results/<model>/<task>/summary.json` (optional `generations.jsonl`, `scatter_*.png`). Figures: `Cond-Gen/results/plots/scaling_{single,double,triple}.png` — single is QED \| SA \| similarity; double is 2×3 (QED, SA, QED+SA/5 × two tag orders); triple is 2×3 (QED, SA, similarity × two orders).

### 3. Docking & Optimization Benchmarks (`PMO-Dock/`)
Protein-aware molecular optimization with ChemLlama + genetic search (`genetic_chemalactica`). Scaling grids live in `PMO-Dock/scale_configs/` (one YAML per task family: hit, lead, spec), in the same style as Cond-Gen: `--model` / `--task(s)` on the CLI, everything else in the file.

Each config uses **seeds 1–5** and **`max_oracle_calls: 10000`**.

| Config | Tasks |
|--------|--------|
| `scale_configs/hit.yaml` | `hit.parp1`, `hit.fa7`, `hit.5ht1b`, `hit.braf`, `hit.jak2` |
| `scale_configs/lead.yaml` | `lead.<protein>_06_<0–2>` (5 proteins × similarity 0.6 × 3 seed ligands) |
| `scale_configs/spec.yaml` | `spec.6nzp_<antitarget>` (`4l00`, `5khw`, `5ut5`) |

Those YAMLs are the scaling grid. The genetic runner still takes `--config_file` (model), `--task_name`, `--seeds`, and `--max_oracle_calls`:

```bash
export PROJECT_ROOT="$PWD/PMO-Dock" PYTHONPATH="$PWD/PMO-Dock:$PYTHONPATH"
python PMO-Dock/genetic_chemalactica/genetic_runner.py \
  --config_file PMO-Dock/genetic_chemalactica/genetic/configs/llama3_170m.yaml \
  --task_name hit.parp1 \
  --reward_type hit \
  --seeds 1 2 3 4 5 \
  --max_oracle_calls 10000
```

Model YAMLs: `llama3_170m.yaml`, `llama3_380m.yaml`, `llama3_1b.yaml`, `llama3_3b.yaml`. Lead needs `benchmark/actives.csv` (seed ligands).

**Open Babel** is required for local docking (QuickVina 3D conformers). Do **not** install it with pip; use conda, and only if you run PMO-Dock docking (skip for Prop-Pred or Cond-Gen):

```bash
conda install -c conda-forge openbabel -y
```

---

## 🛠️ Setup & Installation

### 1. Clone the Repository
```bash
git clone --recurse-submodules git@github.com:YerevaNN/MolScaling.git
cd MolScaling
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

### 2. Environment Setup
Create and activate a conda/mamba environment:
```bash
conda create -n molscaling python=3.10 -y
conda activate molscaling
pip install -r requirements.txt
```

That install includes the PMO-Dock `benchmark` package (`-e ./PMO-Dock`) so Cond-Gen and genetic search can import the shared QED / SA / docking oracles.

### 3. Open Babel (PMO-Dock docking only)
Local docking needs the `openbabel` Python module and the `obabel` binary. They are **not** on PyPI in a form we use; install with conda after the env exists:

```bash
conda install -c conda-forge openbabel -y
```

Skip this step if you only run Prop-Pred or Cond-Gen.

### 4. Key Dependencies
* PyTorch (`>=2.0`)
* Hugging Face `transformers`, `accelerate`
* RDKit (`rdkit`)
* PMO-Dock `benchmark` (`omegaconf`, `flask`, `requests`; Open Babel via conda)
* Polaris Hub SDK (`polaris-lib`)
* Weights & Biases (`wandb`)

---

## 📜 License
This repository is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.
