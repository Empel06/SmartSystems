# src/train.py
import os, glob, numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm

PREP_DIR = "dataset/preprocessed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-3

# Load data
class KwsDataset(Dataset):
    def __init__(self, prep_dir):
        self.x = []
        self.y = []
        self.labels = []
        for i, file in enumerate(sorted(glob.glob(os.path.join(prep_dir, "*.npy")))):
            label = os.path.splitext(os.path.basename(file))[0]
            self.labels.append(label)
            arr = np.load(file)  # (N, n_mels, frames)
            for a in arr:
                self.x.append(a)
                self.y.append(i)
        self.x = np.array(self.x)
        self.y = np.array(self.y)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx]).long()

ds = KwsDataset(PREP_DIR)
num_classes = len(ds.labels)
print("Classes:", ds.labels)
# Simple train/test split
from sklearn.model_selection import train_test_split
indices = list(range(len(ds)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
train_subset = torch.utils.data.Subset(ds, train_idx)
val_subset = torch.utils.data.Subset(ds, val_idx)
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

# Model
class SimpleCNN(nn.Module):
    def __init__(self, in_ch=1, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.net(x)

model = SimpleCNN(in_ch=1, num_classes=num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop
best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    running = 0
    for xb, yb in tqdm(train_loader, desc=f"Train E{epoch+1}"):
        xb = xb.unsqueeze(1).to(DEVICE)  # (B, 1, n_mels, frames)
        yb = yb.to(DEVICE)
        out = model(xb)
        loss = criterion(out, yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        running += loss.item()
    # validate
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.unsqueeze(1).to(DEVICE)
            yb = yb.to(DEVICE)
            out = model(xb)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    acc = correct / total if total>0 else 0
    print(f"Epoch {epoch+1} loss {running/len(train_loader):.4f} val_acc {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        torch.save({
            "model_state": model.state_dict(),
            "labels": ds.labels
        }, os.path.join(MODEL_DIR, "kws_cnn.pt"))
        print("Saved best model.")
print("Training finished. Best acc:", best_acc)
