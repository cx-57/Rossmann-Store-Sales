# Rossmann Store Sales — PyTorch Forecasting

A PyTorch neural-network experiment for predicting daily store sales using historical Rossmann sales data and store characteristics. Code and datasets are organized separately, with a root-level entry point.

## Repository structure

```text
Rossmann-Store-Sales/
├── README.md
├── main.py                 # Run training, validation and prediction
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # Merge data, engineer features, impute and encode
│   └── model.py            # PyTorch Dataset, network, training and prediction
└── data/
    ├── train.csv           # Labeled historical sales
    ├── store.csv           # Store metadata
    └── test.csv            # Unlabeled test features
```

## Setup and run

Requires Python, **NumPy**, **pandas**, **scikit-learn (1.2+)** and **PyTorch**. From the repository root:

```bash
python -m pip install numpy pandas "scikit-learn>=1.2" torch
python main.py
```

`main.py` locates the CSVs relative to the repository, so it does not depend on the shell's current directory. It imports its model and preprocessing logic from `src/`.

## Method

1. Join historical sales and test features with store metadata by store ID. Exclude closed stores from **training** (the original model is trained only on open-store sales).
2. Engineer year, month, day and week-of-year features, impute missing competition-related fields, and use pre-2015 records for training and **2015 records for validation**.
3. One-hot encode store type, assortment and holiday indicators; standardize numeric features. Fit transformations using training data only.
4. Train a **256 → 128 → 64 → 1** feedforward PyTorch network using ReLU, dropout (0.2), Adam (learning rate 0.001), and MSE loss on `log1p(Sales)`. Training uses 20 epochs and batches of 1024.
5. Print training/validation loss each epoch. Apply `expm1` to the test predictions and display the first five predicted sales values.

## Evaluation and limitations

The script reports training and validation loss, but **does not compute a labeled held-out test error or compare performance against a baseline**. The repository's test data do not include sales labels, so no accuracy improvement is claimed. As in the original experiment, the model generates test predictions for available rows without separately forcing closed stores to zero. The validation split is temporal, but no systematic hyperparameter search has been documented.
