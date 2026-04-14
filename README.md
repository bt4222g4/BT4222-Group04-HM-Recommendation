# BT4222 Group 04 — H&M Personalised Fashion Recommendations

This repository contains the source code for the BT4222 project, which builds a fashion recommendation pipeline on the [H&M Personalised Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data) dataset from Kaggle.

---

## Repository Structure

```
BT4222-Group04/
├── BT4222-G4-pipeline.ipynb                 # End-to-end recommendation pipeline
├── BT4222-G4-pipeline (with training).ipynb # Supplementary notebook with training outputs
└── README.md                                # This file
```

> The raw CSV datasets are **not included** in this repository due to file size constraints. See [Datasets](#datasets) below for download instructions.

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
3. Download the three CSV files from Kaggle (see [Datasets](#datasets)) and place them in your Drive at the path above.
4. Run all cells in order: **Runtime → Run all**.
5. Authorise Google Drive access when prompted.

## Datasets

**Automatic handling** — no manual download needed:

1. **Primary**: Notebook auto-downloads from Kaggle using `kagglehub`
2. **Backup**: Shared Google Drive folder:
   [https://drive.google.com/drive/folders/1p-wT4TWw_KuTbZ_1KOY-LetdcN2J2CvB?usp=drive_link](https://drive.google.com/drive/folders/1p-wT4TWw_KuTbZ_1KOY-LetdcN2J2CvB?usp=drive_link)

**Manual download only if automatic methods fail** (Kaggle):
[https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)
(Register/login + **join competition** required)

| File | Size | Description |
|------|------|-------------|
| `articles.csv` | ~36 MB | Metadata for ~105K articles — product type, colour group, department, garment group, and text descriptions |
| `customers.csv` | ~200 MB | Metadata for ~1.4M customers — age, club membership status, and fashion news subscription preference |
| `transactions_train.csv` | ~3.5 GB | ~31M purchase transactions from Sep 2018 to Sep 2020 — customer ID, article ID, price, and sales channel |

After downloading, place all three files in your Google Drive at:
```
/content/drive/MyDrive/BT4222 Group 04/data/
```
If you use a different path, update the `data_path` variable in **Cell 2** of the notebook.

---

## Notes

- **T4 GPU required** for reasonable training time (around 6 hours end-to-end)
- `BT4222-G4-pipeline (with training).ipynb` was added as a supplementary notebook to preserve training and evaluation outputs for verification.
- The supplementary notebook does not change the pipeline design or reported results; it only preserves outputs generated during training and evaluation. The original submission notebook remains the primary source code deliverable, while the later notebook is provided as a supplementary verification artifact to make the training/evaluation results inspectable. 
