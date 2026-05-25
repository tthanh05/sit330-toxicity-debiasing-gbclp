# SIT330 Toxicity Debiasing with GB-CLP

This repository contains the code, results, figures, and final report for the SIT330 / SIT770 High Distinction research task on **robust debiasing for toxicity detection under demographic subpopulation shift and dialect-oriented transfer evaluation**.

The project investigates whether identity-term debiasing methods for toxicity classification remain reliable when demographic subgroup composition changes, and whether fairness gains observed on an identity-based benchmark transfer to a dialect-oriented external evaluation setting.

The main proposed method is:

```text
GB-CLP = Group-Balanced Counterfactual Logit Pairing
```

GB-CLP extends Counterfactual Logit Pairing (CLP) by adding inverse-frequency weighting over label–identity groups.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Research Question](#research-question)
- [Repository Structure](#repository-structure)
- [Methods Compared](#methods-compared)
- [Datasets](#datasets)
- [Main Results](#main-results)
- [Figures](#figures)
- [Report](#report)
- [Environment and Installation](#environment-and-installation)
- [How to Run](#how-to-run)
- [Outputs](#outputs)
- [Reproducibility Notes](#reproducibility-notes)
- [Limitations](#limitations)
- [Ethical and Data Notes](#ethical-and-data-notes)
- [License](#license)

---

## Project Overview

Toxicity detection models are widely used in content moderation, but high average performance can hide unfair behaviour across demographic subgroups. Prior work has shown that toxicity classifiers can assign higher toxicity scores to comments containing demographic identity terms, even when the comment itself is not toxic.

This project studies this issue as a robustness problem. Instead of only asking whether a debiasing method improves fairness on a standard split, the project asks whether debiasing remains useful under demographic subpopulation shift and whether the observed improvements transfer to an external dialect-oriented evaluation setting.

The experimental pipeline compares standard training and counterfactual debiasing methods using:

- CivilComments-WILDS for demographic subpopulation-shift evaluation,
- Davidson hate/offensive-language data for external transfer evaluation,
- DistilBERT as the shared model backbone,
- controlled training and validation protocol,
- overall classification metrics and fairness-oriented metrics.

---

## Research Question

The project is based on the following research question:

> To what extent do debiasing methods for toxicity detection remain effective under demographic subpopulation shift, and do these fairness gains generalise to dialectal variation?

The final report does not claim that the proposed method solves toxicity fairness. Instead, it evaluates whether the method improves specific fairness behaviours under controlled experimental conditions.

---

## Repository Structure

The repository is organised as follows:

```text
sit330-toxicity-debiasing-gbclp/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── SIT330_2_3HD_Toxicity_Debiasing_Notebook.ipynb
│
├── figures/
│   ├── civilcomments_roc_curve_comparison.png
│   ├── civilcomments_max_abs_fpr_gap.png
│   ├── civilcomments_mean_bpsn_auc.png
│   ├── civilcomments_worst_group_acc.png
│   ├── civilcomments_worst_group_auc.png
│   └── dialect_transfer_fpr_gap.png
│
├── results/
│   ├── civilcomments_main_results.csv
│   ├── dialect_transfer_results.csv
│   ├── paper_ready_civilcomments_summary.csv
│   └── training_histories.csv
│
└── report/
    └── SIT330 HD Report.pdf
```

### Root files

| File | Purpose |
|---|---|
| `README.md` | Main project documentation. |
| `LICENSE` | Repository licence. |
| `requirements.txt` | Python dependencies used by the notebook. |
| `.gitignore` | Ignore rules for datasets, checkpoints, raw outputs, and cache files. |

### `notebooks/`

| File | Purpose |
|---|---|
| `SIT330_2_3HD_Toxicity_Debiasing_Notebook.ipynb` | Main experimental notebook. It includes dataset loading, preprocessing, counterfactual generation, model training, evaluation, plotting, dialect-transfer testing, and error analysis. |

### `figures/`

This folder stores the main figures used in the report.

| Figure | Meaning |
|---|---|
| `civilcomments_roc_curve_comparison.png` | ROC curve comparison across ERM, CDA, CLP, and GB-CLP. |
| `civilcomments_max_abs_fpr_gap.png` | Maximum absolute false-positive-rate gap across identity groups. |
| `civilcomments_mean_bpsn_auc.png` | Mean BPSN AUC comparison. |
| `civilcomments_worst_group_acc.png` | Worst-group accuracy comparison. |
| `civilcomments_worst_group_auc.png` | Worst-group AUC comparison. |
| `dialect_transfer_fpr_gap.png` | Exploratory Davidson transfer FPR-gap comparison. |

### `results/`

This folder stores aggregate result tables. These are safe summary outputs and do not include the raw datasets.

| File | Purpose |
|---|---|
| `civilcomments_main_results.csv` | Main CivilComments-WILDS test metrics for all compared methods. |
| `dialect_transfer_results.csv` | Davidson transfer evaluation metrics. |
| `paper_ready_civilcomments_summary.csv` | Compact results table prepared for reporting. |
| `training_histories.csv` | Per-epoch training and validation records. |

### `report/`

| File | Purpose |
|---|---|
| `SIT330 HD Report.pdf` | Final ACM-style research paper submitted for the HD task. |

Only the final PDF report is included in this folder.

---

## Methods Compared

The notebook compares four methods under the same experimental setup.

| Method | Description |
|---|---|
| ERM | Standard empirical risk minimisation using binary cross-entropy. |
| CDA | Counterfactual data augmentation using identity-swapped text. |
| CLP | Counterfactual Logit Pairing, which encourages original and counterfactual logits to be similar. |
| GB-CLP | Proposed method: CLP plus inverse-frequency weighting over label–identity groups. |

### Proposed method: GB-CLP

GB-CLP combines:

1. **Counterfactual identity-term swapping**  
   Identity terms such as `muslim/christian`, `black/white`, and `woman/man` are swapped to create counterfactual text examples.

2. **Counterfactual logit pairing**  
   The model is encouraged to produce similar logits for the original and counterfactual version of the same comment.

3. **Group-balanced weighting**  
   Training examples are weighted using inverse-frequency weights based on label–identity groups.

The method is intentionally simple and transparent, but the report also discusses its limitations.

---

## Datasets

### CivilComments-WILDS

The main benchmark is **CivilComments-WILDS**, accessed using the official `wilds` package. The experiment uses the official train, validation, and test splits rather than a random split, because the project focuses on demographic subpopulation shift.

The controlled subset used in the final experiment is:

```text
Train:      50,000 examples
Validation: 10,000 examples
Test:       20,000 examples
```

Sampling is stratified by:

```text
label
identity_any
```

### Davidson transfer evaluation

The external transfer evaluation uses the Davidson hate-speech and offensive-language dataset through the Hugging Face `datasets` package.

The binary mapping used in the notebook is:

```text
hate speech          -> toxic
offensive language   -> toxic
neither              -> non-toxic
```

The dialect-transfer analysis uses an exploratory lexical proxy grouping. It should not be interpreted as a validated dialect classification result.

---

## Main Results

The main CivilComments-WILDS results are summarised below.

| Method | ROC-AUC | Macro-F1 | Accuracy | Worst-group Acc. | Worst-group AUC | Mean BPSN AUC | Max abs FPR gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| ERM | 0.9382 | 0.8002 | 0.9187 | 0.6168 | 0.8342 | 0.8783 | 0.1151 |
| CDA | 0.9323 | 0.7912 | 0.9184 | 0.5701 | 0.8046 | 0.8595 | 0.1350 |
| CLP | 0.9362 | 0.8001 | 0.9166 | 0.6416 | 0.8209 | 0.8632 | 0.1892 |
| GB-CLP | 0.9365 | 0.7990 | 0.9190 | 0.5599 | 0.8224 | 0.8916 | 0.0807 |

Key observations:

- ERM has the highest ROC-AUC and worst-group AUC.
- CLP has the highest worst-group accuracy.
- GB-CLP has the lowest maximum absolute false-positive-rate gap.
- GB-CLP has the highest mean BPSN AUC.
- No method dominates across all metrics.

The main conclusion is therefore cautious: GB-CLP improves selected fairness measures in the identity-based subpopulation-shift setting, but the improvement is metric-dependent.

---

## Figures

The main figures are stored in `figures/`.

### CivilComments ROC curve

```text
figures/civilcomments_roc_curve_comparison.png
```

This figure compares the ROC curves of ERM, CDA, CLP, and GB-CLP on the CivilComments-WILDS test subset.

### CivilComments FPR gap

```text
figures/civilcomments_max_abs_fpr_gap.png
```

This figure shows the maximum absolute false-positive-rate gap across identity groups. GB-CLP gives the lowest FPR gap in the final experiment.

### Mean BPSN AUC

```text
figures/civilcomments_mean_bpsn_auc.png
```

This figure compares mean BPSN AUC. GB-CLP gives the strongest value on this metric.

### Worst-group metrics

```text
figures/civilcomments_worst_group_acc.png
figures/civilcomments_worst_group_auc.png
```

These figures show that CLP performs best on worst-group accuracy, while ERM performs best on worst-group AUC.

### Davidson transfer FPR gap

```text
figures/dialect_transfer_fpr_gap.png
```

This figure shows the exploratory external transfer FPR-gap result. CDA gives the lowest proxy dialect FPR gap, while GB-CLP does not clearly transfer its CivilComments FPR-gap improvement to this proxy setting.

---

## Report

The final report is stored in:

```text
report/SIT330 HD Report.pdf
```

The report follows the ACM two-column conference format and includes:

1. Introduction
2. Related Work
3. Preliminaries and Problem Definition
4. Proposed Method
5. Experimental Setup
6. Results
7. Discussion
8. Limitations
9. Conclusion and Future Work

The report is written to emphasise evidence-based interpretation. It does not claim that GB-CLP is uniformly better than all baselines. Instead, it discusses the fairness–performance trade-off observed in the results.

---

## Environment and Installation

### Recommended environment

The notebook was designed for Google Colab with GPU acceleration.

Recommended setup:

```text
Runtime: Google Colab
Hardware accelerator: GPU
Preferred GPU: A100 if available
Python: 3.10 or newer
```

The notebook can run locally, but full training is much slower on CPU. For local GPU use, make sure PyTorch is installed with CUDA support.

### Install dependencies

Install dependencies with:

```bash
pip install -r requirements.txt
```

A typical `requirements.txt` should include:

```text
wilds
datasets
transformers
accelerate
evaluate
scikit-learn
pandas
numpy
matplotlib
tqdm
torch
```

If running in Colab, install packages in the first notebook setup cell.

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/sit330-toxicity-debiasing-gbclp.git
cd sit330-toxicity-debiasing-gbclp
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the notebook

```text
notebooks/SIT330_2_3HD_Toxicity_Debiasing_Notebook.ipynb
```

### 4. Run the notebook sections in order

The notebook should be run from top to bottom. The main stages are:

1. Install/import packages
2. Set configuration
3. Load CivilComments-WILDS
4. Preprocess and sample data
5. Generate counterfactual identity-swapped text
6. Build group weights
7. Define dataset, dataloader, model, and metrics
8. Train ERM, CDA, CLP, and GB-CLP
9. Evaluate on CivilComments-WILDS
10. Plot CivilComments results
11. Run Davidson transfer evaluation
12. Run error analysis
13. Export paper-ready results

### 5. Adjust runtime settings if needed

The final experiment uses:

```python
N_TRAIN = 50000
N_VAL = 10000
N_TEST = 20000
N_DIALECT = 10000
EPOCHS = 10
BATCH_SIZE = 64
```

For a quick test run, reduce the dataset sizes or enable quick mode if available in the notebook.

---

## Outputs

The notebook writes outputs to a project output directory and selected outputs are included in this repository.

Typical generated outputs include:

```text
civilcomments_main_results.csv
dialect_transfer_results.csv
paper_ready_civilcomments_summary.csv
training_histories.csv
civilcomments_roc_curve_comparison.png
civilcomments_max_abs_fpr_gap.png
civilcomments_mean_bpsn_auc.png
civilcomments_worst_group_acc.png
civilcomments_worst_group_auc.png
dialect_transfer_fpr_gap.png
```

Some raw diagnostic files may contain original comments or tweets. These are intentionally not included in the repository.

---

## Reproducibility Notes

The notebook controls the main experimental conditions by using:

- the same model backbone across all methods,
- the same tokenizer,
- the same CivilComments-WILDS official split structure,
- the same optimiser and learning-rate schedule,
- validation-based checkpoint selection,
- validation-only threshold tuning,
- the same evaluation metrics across methods.

The final reported experiment uses one random seed. Because of this, the report does not claim statistical significance or confidence intervals.

---

## Limitations

Important limitations:

1. The experiment uses a controlled subset of CivilComments-WILDS rather than the full benchmark.
2. The reported results use one random seed.
3. The Davidson transfer evaluation uses an exploratory proxy grouping rather than a validated dialect classifier.
4. Identity-term swapping is simple and may create unnatural counterfactual text.
5. The experiment uses DistilBERT only; results may differ with larger or newer transformer models.
6. GB-CLP balances label–identity groups based on whether any tracked identity is present, not every individual identity group separately.

These limitations are discussed in the report.

---

## Ethical and Data Notes

This repository does not redistribute CivilComments-WILDS or Davidson dataset contents. The datasets are downloaded from their original sources through their respective Python packages or dataset interfaces.

The repository also avoids committing raw error-analysis files that may contain toxic text, offensive language, or user-generated comments. Only aggregate result tables and figures are intended to be included.

The models and analysis are for academic research and coursework purposes. They should not be used directly for real-world moderation decisions without further validation, human oversight, and a stronger fairness and safety evaluation.

---

## License

This repository is released under the MIT License.

The MIT License applies to the code and documentation written for this project. External datasets, pretrained models, and third-party libraries remain subject to their original licences and terms of use.
