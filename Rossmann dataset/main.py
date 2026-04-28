import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


# --- PREPROCESSING ---


# Load CSVs 
train_df = pd.read_csv("train.csv")
store_df = pd.read_csv("store.csv")
test_df  = pd.read_csv("test.csv")


# Merge store dataset with train and test dataset
train_df = train_df.merge(store_df, on="Store", how="left")
test_df  = test_df.merge(store_df,  on="Store", how="left")


# Clean data
train_df = train_df[train_df["Open"] == 1]   # Drop closed stores
train_df = train_df.drop(columns=["Customers", "Open"])
test_df["Open"].fillna(1, inplace=True)
test_df  = test_df.drop(columns=["Open"])


# Convert month and dates to more readable numbers 
for df in [train_df, test_df]:
    df["Date"]       = pd.to_datetime(df["Date"])
    df["Year"]       = df["Date"].dt.year
    df["Month"]      = df["Date"].dt.month
    df["Day"]        = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week
    df = df.drop(columns=["Date"])


# Fill in missing values 
fill_with_median = ["CompetitionDistance"]
fill_with_zero   = ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
                    "Promo2SinceWeek", "Promo2SinceYear"]

for col in fill_with_median:
    median = train_df[col].median()
    train_df[col] = train_df[col].fillna(median)
    test_df[col]  = test_df[col].fillna(median)

for col in fill_with_zero:
    train_df[col] = train_df[col].fillna(0)
    test_df[col]  = test_df[col].fillna(0)


# Split data to before and after 2015
val_df   = train_df[train_df["Year"] == 2015].copy()
train_df = train_df[train_df["Year"] != 2015].copy()


# Encode categorical columns and normalize numerical columns 
train_df["StateHoliday"] = train_df["StateHoliday"].astype(str)
test_df["StateHoliday"]  = test_df["StateHoliday"].astype(str)

cat_cols = ["StoreType", "Assortment", "StateHoliday"]
num_cols = ["Store", "DayOfWeek", "Promo", "CompetitionDistance",
            "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
            "Promo2", "Promo2SinceWeek", "Promo2SinceYear",
            "SchoolHoliday", "Year", "Month", "Day", "WeekOfYear"]


# OneHot Encoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
train_cat = encoder.fit_transform(train_df[cat_cols])
val_cat   = encoder.transform(val_df[cat_cols])
test_cat  = encoder.transform(test_df[cat_cols])


# Standard Scaler 
scaler    = StandardScaler()
train_num = scaler.fit_transform(train_df[num_cols])
val_num   = scaler.transform(val_df[num_cols])
test_num  = scaler.transform(test_df[num_cols])

X_train = np.hstack([train_cat, train_num])
X_val   = np.hstack([val_cat,   val_num])
X_test  = np.hstack([test_cat,  test_num])

y_train = train_df["Sales"].values
y_val   = val_df["Sales"].values



# --- BUILDING MODEL ---



# Dataset class
class RossmannDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(np.log1p(y), dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# Neural Network
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
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# Train function
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Evaluate model 
@torch.no_grad()    # Stop gradient from running during evaluation
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        total_loss += criterion(model(X_batch), y_batch).item()
    return total_loss / len(loader)


# Main function
def main(X_train, y_train, X_val, y_val):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(RossmannDataset(X_train, y_train), batch_size=1024, shuffle=True)
    val_loader   = DataLoader(RossmannDataset(X_val, y_val),     batch_size=1024)

    model     = SalesNet(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(1, 21):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss   = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return model


# Run Model 
model = main(X_train, y_train, X_val, y_val)


# Predict
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

model.eval()
with torch.no_grad():
    log_preds   = model(X_test_t)
    sales_preds = np.expm1(log_preds.cpu().numpy())  # reverse the log1p

print("Sample predictions:", sales_preds[:5])
