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
EPOCHS = 100  # Increased, but early stopping will prevent overfitting
LR = 1e-3
PATIENCE = 5  # Stop if no improvement for 5 epochs


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
print(f"Total samples: {len(ds)}")

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


# Training loop with Early Stopping
best_val_acc = 0.0
patience_counter = 0

print(f"\nStarting training (early stopping with patience={PATIENCE})...\n")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    for xb, yb in tqdm(train_loader, desc=f"Train E{epoch+1}"):
        xb = xb.unsqueeze(1).to(DEVICE)  # (B, 1, n_mels, frames)
        yb = yb.to(DEVICE)
        out = model(xb)
        loss = criterion(out, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Validate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.unsqueeze(1).to(DEVICE)
            yb = yb.to(DEVICE)
            out = model(xb)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    
    val_acc = correct / total if total > 0 else 0
    avg_loss = running_loss / len(train_loader)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    # Early Stopping: save only if accuracy improves
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save({
            "model_state": model.state_dict(),
            "labels": ds.labels
        }, os.path.join(MODEL_DIR, "kws_cnn.pt"))
        print(f"  -> Best model saved! (accuracy improved to {best_val_acc:.4f})")
    else:
        patience_counter += 1
        print(f"  -> No improvement ({patience_counter}/{PATIENCE})")
        
        # Stop early if no improvement for PATIENCE epochs
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered! No improvement for {PATIENCE} epochs.")
            print(f"Best validation accuracy: {best_val_acc:.4f}")
            break

print(f"\nTraining finished. Best accuracy: {best_val_acc:.4f}")