# DeepSemantics at SemEval-2026 Task 9: Label-Wise Optimization with Adaptive Focal Loss for Polarization Manifestation Identification

> **Team:** DeepSemantics — African Institute for Mathematical Sciences (AIMS), South Africa  
> **Competition:** [SemEval-2026 Task 9 — Detecting Multilingual Online Polarization (POLAR)](https://semeval.github.io/)  
> **Subtask:** Subtask 3 — Manifestation Identification (English & Hausa)

---

## 📄 Abstract

This repository contains the code for our system submitted to **SemEval-2026 Task 9, Subtask 3: Polarization Manifestation Identification**. We address fine-grained multi-label classification of polarization manifestations (Vilification, Dehumanization, Extreme Language, Lack of Empathy, Invalidation, Stereotype) in English and Hausa social media texts.

Our approach combines:
- **Transformer encoders** (RoBERTa-base for English, Afro-XLM-R-small for Hausa)
- **One-vs-Rest (OvR) framework** for label-wise modeling
- **Adaptive Focal Loss** (English) and **Weighted Binary Cross-Entropy** (Hausa)
- **Controlled oversampling** with Easy Data Augmentation (EDA)
- **Label-wise threshold optimization** via stratified K-fold validation

**Official test results:**
| Language | Macro-F1 | Leaderboard Rank |
|----------|----------|-----------------|
| English  | 0.464    | 14th            |
| Hausa    | 0.192    | 5th             |

---

## 📂 Repository Structure

```
├── substack_3_final.ipynb   # Full pipeline: preprocessing, training, evaluation, error analysis
└── README.md
```

The notebook contains the entire pipeline end-to-end:
- Data loading and exploratory analysis
- Controlled oversampling + Easy Data Augmentation (EDA)
- One-vs-Rest training with Adaptive Focal Loss (EN) / Weighted BCE (HA)
- Label-wise threshold optimization via stratified K-fold
- Evaluation and error analysis

---

## 📊 Data

Data comes from the **POLAR SemEval-2026 shared task**. Download the dataset from the official repository:

🔗 **[Polar-SemEval/data-public](https://github.com/Polar-SemEval/data-public/)**

Once downloaded, update the data paths at the top of the notebook accordingly.

### Label Distribution

The dataset is highly imbalanced across all six manifestation labels:

| Manifestation    | English (% Present) | Hausa (% Present) |
|------------------|---------------------|-------------------|
| Stereotype       | 15.1%               | 4.3%              |
| Vilification     | 26.6%               | 1.2%              |
| Dehumanization   | 12.1%               | 3.5%              |
| Extreme Language | 23.9%               | 3.0%              |
| Lack of Empathy  | 11.1%               | 0.9%              |
| Invalidation     | 18.2%               | 0.2%              |

---

## ⚙️ Installation

```bash
git clone https://github.com/TIAO-Eliasse/nlp_project.git
cd nlp_project
pip install -r requirements.txt
```

**Main dependencies:**
- Python 3.8+
- PyTorch
- Transformers (HuggingFace)
- scikit-learn
- iterstrat
- nlpaug

The notebook was developed and run on **Google Colab** (free tier). You can open it directly there:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TIAO-Eliasse/nlp_project/blob/main/substack_3_final.ipynb)

---

## ⚙️ Model Configurations

| Parameter          | English             | Hausa              |
|--------------------|---------------------|--------------------|
| Model              | `roberta-base`      | `afro-xlmr-small`  |
| Learning Rate      | 1e-5                | 1e-5               |
| Batch Size         | 4                   | 4                  |
| Grad. Accumulation | 1                   | 2                  |
| Epochs             | 5                   | 3                  |
| Max Length         | 512                 | 512                |
| Loss               | Focal (γ=2, α=0.25) | Weighted BCE       |
| Oversampling       | —                   | ratio > 5          |
| Threshold Search   | [0.01 – 0.99]       | [0.01 – 0.99]      |

---

## 📈 Results

### Validation Set

| Label            | EN — OvR | EN — Multi-label | HA — OvR |
|------------------|----------|------------------|----------|
| Stereotype       | 0.560    | 0.420            | 0.286    |
| Vilification     | 0.590    | 0.580            | 0.111    |
| Dehumanization   | 0.430    | 0.380            | 0.364    |
| Extreme Language | 0.570    | 0.300            | 0.375    |
| Lack of Empathy  | 0.360    | 0.000            | 0.057    |
| Invalidation     | 0.530    | 0.290            | 0.000    |
| **Macro-F1**     | **0.500**| **0.330**        | **0.200**|

### Official Test Set

| Label            | English F1 | Hausa F1 |
|------------------|------------|----------|
| Stereotype       | 0.420      | 0.297    |
| Vilification     | 0.625      | 0.165    |
| Dehumanization   | 0.378      | 0.252    |
| Extreme Language | 0.564      | 0.185    |
| Lack of Empathy  | 0.356      | 0.102    |
| Invalidation     | 0.442      | 0.154    |
| **Macro-F1**     | **0.464**  | **0.192**|

### Optimized Decision Thresholds

| Manifestation    | English | Hausa |
|------------------|---------|-------|
| Vilification     | 0.18    | 0.990 |
| Extreme Language | 0.41    | 0.990 |
| Stereotype       | 0.57    | 0.900 |
| Invalidation     | 0.21    | 0.500 |
| Lack of Empathy  | 0.07    | 0.010 |
| Dehumanization   | 0.31    | 0.090 |

---

## 📜 Citation

If you use this code, please cite:

```bibtex
@inproceedings{tiao2026deepsemantics,
  title     = {DeepSemantics at SemEval-2026 Task 9: Label-Wise Optimization with Adaptive Focal Loss for Polarization Manifestation Identification},
  author    = {Tiao, Eliasse and Edou, Josue R. and Gohouede, Mahugnon A. L.},
  booktitle = {Proceedings of the 20th International Workshop on Semantic Evaluation (SemEval-2026)},
  year      = {2026},
  organization = {Association for Computational Linguistics}
}
```

---

## 🙏 Acknowledgements

We thank **Shamsuddeen Hassan Muhammad** and **Idris Abdulmumin** for their NLP course and for encouraging us to participate in the POLAR 2026 shared task. We also thank the task organizers for providing the POLAR dataset and benchmark.

---

## 📬 Contact

| Name                     | Email                  |
|--------------------------|------------------------|
| Eliasse Tiao             | eliasse@aims.ac.za     |
| Josue R. Edou            | josue@aims.ac.za       |
| Mahugnon A. L. Gohouede  | aimeloick@aims.ac.za   |

**Institution:** African Institute for Mathematical Sciences (AIMS), South Africa
