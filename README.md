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
Framework for steering autoregressive molecular language models to generate novel chemical structures subject to specific target properties, substructures, or multi-objective constraints.

### 3. Docking & Optimization Benchmarks (`PMO-Dock/`)
Benchmarking suite integrating structure-based molecular docking tools with molecular optimization loops.

---

## 🛠️ Setup & Installation

### 1. Clone the Repository
```bash
git clone git@github.com:YerevaNN/MolScaling.git
cd MolScaling
```

### 2. Environment Setup
Create and activate a conda/mamba environment:
```bash
conda create -n molscaling python=3.10 -y
conda activate molscaling
pip install -r requirements.txt
```

### 3. Key Dependencies
* PyTorch (`>=2.0`)
* Hugging Face `transformers`, `accelerate`
* RDKit (`rdkit`)
* Polaris Hub SDK (`polaris-lib`)
* Weights & Biases (`wandb`)

---

## 📜 License
This repository is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.
