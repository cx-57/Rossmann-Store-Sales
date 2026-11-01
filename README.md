# Rossmann Store Sales — PyTorch Forecasting

A PyTorch neural-network experiment for predicting daily store sales using historical Rossmann sales data and store characteristics. The original single-file script is now split into preprocessing, model/training and a main entry point.

## Project structure

```text
Rossmann-Store-Sales/
├── README.md
└── Rossmann dataset/
    ├── train.csv
    ├── store.csv
    ├── test.csv
    ├── preprocessing.py   # Merge, feature engineering, imputation and encoding
    ├── model.py           # Dataset, neural network, training and prediction
    └── main.py            # Runnable entry point
```

## Setup and run

Requires Python, **NumPy**, **pandas**, **scikit-learn (1.2+)** and **PyTorch**.

```bash
python -m pip install numpy pandas "scikit-learn>=1.2" torch
python "Rossmann dataset/main.py"
```

The script locates its CSV files relative to `main.py`; it can be launched from the repository root.

## Method

1. Join `train.csv` and `test.csv` with `store.csv` by store ID. Exclude closed stores from **training**, because the original model is trained on open-store sales only.
2. Engineer year, month, day and week-of-year features, impute competition-related fields and split historical training records into training (pre-2015) and **2015 validation**.
3. One-hot encode store type, assortment and holiday indicators; standardize numeric features. Fit these transforms on the training split only.
4. Train a **256 → 128 → 64 → 1** PyTorch network with ReLU activations, 0.2 dropout, Adam (learning rate `0.001`), and MSE loss on `log1p(Sales)`. The script runs **20 epochs** with batch size 1024.
5. Print training/validation loss each epoch; transform test predictions back into sales units with `expm1` and print a sample.

## Evaluation and limitations

The training run reports losses but **does not compute a held-out test error or compare against a baseline**. The public test labels are not in this repository, so a forecast-accuracy claim would be premature. As in the original experiment, test predictions are produced for the available rows without post-processing closed-store forecasts. The 2015 split is temporal, but hyperparameters were not selected through a systematic tuning study.
