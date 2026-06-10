"""
Fine-tune ResNet18 binary classifier: real vs synthetic spectrogram.

Dataset  : datasets/spectrograms/  (genuine_*.png = 0, manipulated_*.png = 1)
Split    : 80% train / 20% validation
Model    : ResNet18 pretrained on ImageNet, final layer replaced with binary head
Training : CPU-friendly — ~20 epochs, small batch size
Output   : models/spectrogram_cnn.pt
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import random, csv

SPEC_DIR   = Path("/Users/salonibhardwaj/Desktop/Noise /datasets/spectrograms")
MODEL_DIR  = Path("/Users/salonibhardwaj/Desktop/Noise /models")
MODEL_PATH = MODEL_DIR / "spectrogram_cnn.pt"
MODEL_DIR.mkdir(exist_ok=True)

EPOCHS     = 20
BATCH_SIZE = 8
LR         = 1e-4
SEED       = 42
random.seed(SEED)
torch.manual_seed(SEED)


# ── Dataset ───────────────────────────────────────────────────────────────────
class SpectrogramDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img   = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


# ── Transforms ────────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Load all images and labels ─────────────────────────────────────────────────
genuine_imgs    = sorted(SPEC_DIR.glob("genuine_*.png"))
manipulated_imgs = sorted(SPEC_DIR.glob("manipulated_*.png"))

all_paths  = genuine_imgs + manipulated_imgs
all_labels = [0] * len(genuine_imgs) + [1] * len(manipulated_imgs)

print(f"Dataset: {len(genuine_imgs)} genuine (0) + {len(manipulated_imgs)} manipulated (1)")
print(f"Total  : {len(all_paths)} images\n")

# Shuffle and split 80/20
combined = list(zip(all_paths, all_labels))
random.shuffle(combined)
all_paths, all_labels = zip(*combined)

split      = int(0.8 * len(all_paths))
train_paths, val_paths   = all_paths[:split], all_paths[split:]
train_labels, val_labels = all_labels[:split], all_labels[split:]

print(f"Train : {len(train_paths)} images")
print(f"Val   : {len(val_paths)} images\n")

train_ds = SpectrogramDataset(train_paths, train_labels, train_tf)
val_ds   = SpectrogramDataset(val_paths,   val_labels,   val_tf)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)


# ── Model — ResNet18 with binary head ─────────────────────────────────────────
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)   # binary: real=0, synthetic=1

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)


# ── Training loop ─────────────────────────────────────────────────────────────
print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}")
print("-" * 50)

best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for imgs, labels in train_dl:
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss    += loss.item() * imgs.size(0)
        preds          = outputs.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total   += imgs.size(0)

    scheduler.step()

    # Validate
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_dl:
            outputs   = model(imgs)
            loss      = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            preds     = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total   += imgs.size(0)

    t_loss = train_loss / train_total
    t_acc  = train_correct / train_total * 100
    v_loss = val_loss / val_total
    v_acc  = val_correct / val_total * 100

    flag = "  ← best" if v_acc > best_val_acc else ""
    print(f"  {epoch:>3}  {t_loss:>10.4f}  {t_acc:>8.1f}%  {v_loss:>8.4f}  {v_acc:>6.1f}%{flag}")

    if v_acc > best_val_acc:
        best_val_acc = v_acc
        torch.save(model.state_dict(), MODEL_PATH)

print(f"\nBest val accuracy : {best_val_acc:.1f}%")
print(f"Model saved       : {MODEL_PATH}")


# ── Final evaluation on validation set ────────────────────────────────────────
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

print(f"\n{'File':<55}  {'Pred':<12}  {'True':<12}  Correct")
print("-" * 90)

tp = tn = fp = fn = 0
with torch.no_grad():
    for path, label in zip(val_paths, val_labels):
        img   = val_tf(Image.open(path).convert("RGB")).unsqueeze(0)
        out   = model(img)
        prob  = torch.softmax(out, dim=1)[0]
        pred  = out.argmax(dim=1).item()
        pred_name = "real" if pred == 0 else "synthetic"
        true_name = "real" if label == 0 else "synthetic"
        correct   = "✓" if pred == label else "✗"

        if pred == 1 and label == 1: tp += 1
        elif pred == 0 and label == 0: tn += 1
        elif pred == 1 and label == 0: fp += 1
        else: fn += 1

        print(f"  {Path(path).name:<53}  {pred_name:<12}  {true_name:<12}  {correct}  "
              f"(real={prob[0]:.2f}, synth={prob[1]:.2f})")

total = tp + tn + fp + fn
print(f"\n{'='*60}")
print(f"  Accuracy   : {(tp+tn)/total*100:.1f}%")
print(f"  True  real (TN) : {tn}   |  False synth (FP): {fp}")
print(f"  True synth (TP) : {tp}   |  False real  (FN): {fn}")
print(f"{'='*60}")
