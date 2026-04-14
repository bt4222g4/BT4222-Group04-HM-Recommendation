# BT4222 Group 04 — H&M Personalised Fashion Recommendations

This repository contains the source code and datasets for the BT4222 project, which builds a fashion recommendation pipeline on the [H&M Personalised Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) dataset from Kaggle.

---

## Repository Structure

```
BT4222-Group04/
├── Group_04_Pipeline.ipynb       # Main pipeline notebook (see below)
├── README.md                     # This file
└── data/
    ├── articles.csv              # Article metadata (from Kaggle)
    ├── customers.csv             # Customer metadata (from Kaggle)
    └── transactions_train.csv    # Transaction history (from Kaggle)
```

> **Note:** The `data/` folder contains only the three raw CSV files. Cached artefacts (`.npy`, `.pkl`, `.pt`) are generated automatically at runtime and stored to the same `data_path`. See [Cached Artefacts](#cached-artefacts) below.

---

## Pipeline Overview

`Group_04_Pipeline.ipynb` runs an end-to-end recommendation pipeline in five steps:

| Step | Description |
|------|-------------|
| 1 | **Data Cleaning** — maps IDs to INT32, cleans transactions, customers, and articles |
| 2 | **Feature Engineering** — cyclic time features, transaction-based, article-based, and customer-based features |
| 3 | **NCF Model Training** — expanding-window walk-forward validation across the training year |
| 4 | **Inference** — warm-start (NCF scoring) and cold-start (candidate generation via global trends + NN similarity) |
| 5 | **Evaluation** — MAP@10 segmented by warm/cold-start status |

---

## Running Environment

**Platform:** Google Colaboratory (Colab)  
**Runtime:** Python 3 — GPU recommended (T4 or better) for Steps 3 and 4

### Step-by-step Instructions

1. Upload `Group_04_Pipeline.ipynb` to Google Colab, or open it directly from Google Drive.
2. Set runtime to GPU: **Runtime → Change runtime type → T4 GPU**.
3. Upload the three CSV files from `data/` to your Google Drive at the following path:
   ```
   /content/drive/MyDrive/BT4222 Group 04/data/
   ```
   If you use a different path, update the `data_path` variable in **Cell 2** accordingly.
4. Run all cells in order: **Runtime → Run all**.
5. Authorise Google Drive access when prompted.

### Dependencies

All packages below are pre-installed in Google Colab — no `pip install` steps are required.

| Package | Version (Colab default) | Purpose |
|---------|------------------------|---------|
| `numpy` | ≥ 1.24 | Numerical computing |
| `pandas` | ≥ 1.5 | Data manipulation |
| `matplotlib` | ≥ 3.6 | Visualisation |
| `scikit-learn` | ≥ 1.2 | Preprocessing, TF-IDF, Nearest Neighbours |
| `scipy` | ≥ 1.10 | Sparse matrix operations |
| `torch` (PyTorch) | ≥ 2.0 | NCF model and autoencoder training |
| `tqdm` | ≥ 4.0 | Progress bars |

> If running **outside Colab**, install the above via `pip` and replace the Google Drive mount cell (Cell 1) with a local file path.

---

## Cached Artefacts

To avoid recomputing expensive operations on repeated runs, the pipeline saves and loads the following artefacts to/from `data_path`:

| File | Description |
|------|-------------|
| `stage2_expanding_window_model.pt` | Trained NCF model weights | 
| `item_autoencoder.pt` | Autoencoder weights for item embeddings | 
| `item_matrix.npy` | Raw dense item feature matrix | 
| `latent_matrix.npy` | 64-dim L2-normalised latent embeddings | 
| `item_matrix_article_ids.csv` | Article ID index for the item matrix | 
| `neighbours_ae.pkl` | Pre-computed top-50 cosine NN dict |

**These files are not included in this repository.** They will be created automatically the first time the notebook runs with `NN_TRAIN = True`. On subsequent runs, set `NN_TRAIN = False` (the default) to load from cache instead of rebuilding.

To force a full rebuild from scratch, set `NN_TRAIN = True` in Cell 55.

---

## Datasets

The raw datasets are sourced from the Kaggle competition:  
**[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)**

| File | Description |
|------|-------------|
| `articles.csv` | Metadata for ~105K articles (product type, colour, department, etc.) |
| `customers.csv` | Metadata for ~1.4M customers (age, club membership, etc.) |
| `transactions_train.csv` | ~31M purchase transactions from Sep 2018 to Sep 2020 |

> The datasets are available in this repository and via the Kaggle link above.

---

## Notes

- Reported model performance (MAP@10) can be reproduced by running the notebook end-to-end with `NN_TRAIN = True`. 
