import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

csv_path = "data.csv"
data_df = pd.read_csv(csv_path)

torch.manual_seed(128)

print(data_df.head())


class HeartDataset:
    def __init__(self, file):
        df = pd.read_csv(file)
        df_cleaned = df.drop("education", axis=1)
        df_cleaned = df_cleaned.dropna()
        if len(df_cleaned) != len(df):
            print("null values found and removed")

        x = df_cleaned.iloc[:, 0:14].values
        y = df_cleaned.iloc[:, 14].values
        sc = StandardScaler()
        x_train = sc.fit_transform(x)
        y_train = y

        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.Y_train = torch.tensor(y_train)

    def __len__(self):
        return len(self.Y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.Y_train[idx]


dataset = HeartDataset(csv_path)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

BATCH_SIZE = 16
NUM_FEATURES = 14

train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

device = "cpu"


class LogRegModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(14, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        logits = self.relu(x)

        return logits


model = LogRegModel().to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-3)
total_params = sum(p.numel() for p in model.parameters())
print("Total parameters:", total_params)

for X, Y in val_dataloader:
    print(f"Shape of X: {X.shape}")
    print(f"Shape of Y: {Y.shape} {Y.dtype}")
    x = X.to(device)
    y = Y.to(device)
    break
pred = model(x)


def train(dataloader, model, loss_fn, optimizer, train_losses):
    size = len(dataloader.dataset)
    model.train()
    loss_sum = 0
    num_batches = len(dataloader)
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_func(pred, y)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)

    mean_loss = loss_sum / num_batches
    train_losses.append(mean_loss)


def test(dataloader, model, loss_fn, val_losses):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    val_losses.append(test_loss)


test_losses = []
val_losses = []
for i in range(70):
    train(train_dataloader, model, loss_func, optimizer, test_losses)
    test(val_dataloader, model, loss_func, val_losses)

print("Finished Training")

if os.path.exists("model.pth"):
    os.remove("model.pth")

torch.save(model.state_dict(), "model.pth")
