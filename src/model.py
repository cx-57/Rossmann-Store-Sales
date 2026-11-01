"""PyTorch model, training loop and prediction for Rossmann sales."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class RossmannDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = (
            torch.tensor(np.log1p(y), dtype=torch.float32)
            if y is not None
            else None
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx]) if self.y is not None else self.X[idx]


class SalesNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        total_loss += criterion(model(X_batch), y_batch).item()
    return total_loss / len(loader)


def train_model(X_train, y_train, X_val, y_val):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        RossmannDataset(X_train, y_train), batch_size=1024, shuffle=True
    )
    val_loader = DataLoader(
        RossmannDataset(X_val, y_val), batch_size=1024
    )
    model = SalesNet(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(1, 21):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )
    return model


@torch.no_grad()
def predict_sales(model, X_test):
    model.eval()
    device = next(model.parameters()).device
    inputs = torch.tensor(X_test, dtype=torch.float32, device=device)
    log_predictions = model(inputs)
    return np.expm1(log_predictions.cpu().numpy())
