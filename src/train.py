# src/train.py
import os, glob, numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Configuraties (Globale variabelen zijn OK) ---
PREP_DIR = "dataset/preprocessed"
MODEL_DIR = "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 150
LR = 0.0005
PATIENCE = 10

# --- Definities (Deze moeten beschikbaar zijn voor import) ---

class KwsDataset(Dataset):
    def __init__(self, prep_dir):
        self.x = []
        self.y = []
        self.labels = []
        
        expected_shape = None
        files = sorted(glob.glob(os.path.join(prep_dir, "*.npy")))
        
        for i, file in enumerate(files):
            label = os.path.splitext(os.path.basename(file))[0]
            self.labels.append(label)
            arr = np.load(file)  # (N, n_mels, frames)
            
            if expected_shape is None:
                expected_shape = arr[0].shape
                # Print alleen als we direct runnen, niet bij import
                if __name__ == "__main__":
                    print(f"Expected shape per spectrogram: {expected_shape}")
            
            for a in arr:
                if a.shape != expected_shape:
                    if a.shape[0] == expected_shape[0]:
                        frames_expected = expected_shape[1]
                        frames_actual = a.shape[1]
                        
                        if frames_actual < frames_expected:
                            a = np.pad(a, ((0,0), (0, frames_expected - frames_actual)))
                        else:
                            a = a[:, :frames_expected]
                
                self.x.append(a)
                self.y.append(i)
        
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        
        if __name__ == "__main__":
            print(f"Loaded {len(self.x)} samples with shape {self.x.shape}")

    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx]).long()

class SimpleCNN(nn.Module):
    def __init__(self, in_ch=1, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1: 32 filters
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2: 64 filters
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3: 128 filters (NIEUW)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Head
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(0.4),  # Hoger (0.4) tegen overfitting
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# --- De "Main Guard" ---

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load data
    ds = KwsDataset(PREP_DIR)
    num_classes = len(ds.labels)
    print("Classes:", ds.labels)
    print(f"Total samples: {len(ds)}")

    # 2. Train/test split
    indices = list(range(len(ds)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_subset = torch.utils.data.Subset(ds, train_idx)
    val_subset = torch.utils.data.Subset(ds, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Model setup
    model = SimpleCNN(in_ch=1, num_classes=num_classes).to(DEVICE)
    # Bereken gewichten op basis van counts
    class_counts = np.bincount(ds.y)  # [280, 280, 280, 280, 280, 200]
    weights = 1.0 / class_counts
    weights = weights / weights.sum() # Normaliseren
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 4. Training loop
    best_val_acc = 0.0
    patience_counter = 0

    print(f"\nStarting training (early stopping with patience={PATIENCE})...\n")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0
        for xb, yb in tqdm(train_loader, desc=f"Train E{epoch+1}"):
            xb = xb.unsqueeze(1).to(DEVICE)
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
        
        # Per-class accuracy
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.unsqueeze(1).to(DEVICE)
                yb = yb.to(DEVICE)
                out = model(xb)
                preds = out.argmax(dim=1)
                for i, label_idx in enumerate(yb):
                    label_idx = label_idx.item()
                    class_total[label_idx] += 1
                    if preds[i] == label_idx:
                        class_correct[label_idx] += 1
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")
        for i, label in enumerate(ds.labels):
            class_acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            print(f"    {label}: {class_acc:.2%}")

        
        # Early Stopping
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
            
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping triggered! No improvement for {PATIENCE} epochs.")
                break

    print(f"\nTraining finished. Best accuracy: {best_val_acc:.4f}")
