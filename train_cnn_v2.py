"""
Retrain ResNet18 with both noise-removal AND noise-addition spectrograms.

Dataset:
  datasets/spectrograms/          genuine_*.png        → label 0  (real)
  datasets/spectrograms/          manipulated_*.png    → label 1  (noise removal)
  datasets/spectrograms_noise_addition/  noise_addition_*.png → label 1  (noise addition)

Why v2:
  v1 trained only on noise removal (dark stripe).
  v2 trains on both dark stripe + bright stripe so the CNN detects
  either manipulation type at full-clip level.

Output: models/spectrogram_cnn_v2.pt
"""

import torch, torch.nn as nn, random, csv
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

BASE               = Path(os.environ.get("NOISE_BASE_DIR", str(Path(__file__).parent)))
SPEC_DIR           = BASE / "datasets" / "spectrograms"
SPEC_DIR_ADDITION  = BASE / "datasets" / "spectrograms_noise_addition"
MODEL_DIR          = BASE / "models"
MODEL_PATH         = MODEL_DIR / "spectrogram_cnn_v2.pt"
MODEL_DIR.mkdir(exist_ok=True)

EPOCHS     = 20
BATCH_SIZE = 8
LR         = 1e-4
SEED       = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ── Collect all images ────────────────────────────────────────────────────────
genuine_imgs   = sorted(SPEC_DIR.glob("genuine_*.png"))
removal_imgs   = sorted(SPEC_DIR.glob("manipulated_*.png"))
addition_imgs  = sorted(SPEC_DIR_ADDITION.glob("noise_addition_*.png"))

all_paths  = genuine_imgs  + removal_imgs  + addition_imgs
all_labels = ([0] * len(genuine_imgs) +
              [1] * len(removal_imgs) +
              [1] * len(addition_imgs))

print(f"Dataset breakdown:")
print(f"  Genuine (real=0)          : {len(genuine_imgs)}")
print(f"  Noise removal (synth=1)   : {len(removal_imgs)}")
print(f"  Noise addition (synth=1)  : {len(addition_imgs)}")
print(f"  Total                     : {len(all_paths)}")
print(f"  Class balance  real:synth = {len(genuine_imgs)}:{len(removal_imgs)+len(addition_imgs)}\n")

# ── Train / val split ─────────────────────────────────────────────────────────
combined = list(zip(all_paths, all_labels))
random.shuffle(combined)
all_paths, all_labels = zip(*combined)

split        = int(0.8 * len(all_paths))
train_paths  = list(all_paths[:split]);   train_labels = list(all_labels[:split])
val_paths    = list(all_paths[split:]);   val_labels   = list(all_labels[split:])

print(f"Train : {len(train_paths)} images")
print(f"Val   : {len(val_paths)} images\n")

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

# ── Dataset ───────────────────────────────────────────────────────────────────
class SpectrogramDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]

# ── Weighted sampler to handle class imbalance (100 real vs 799 synthetic) ───
n_real    = train_labels.count(0)
n_synth   = train_labels.count(1)
w_real    = 1.0 / n_real
w_synth   = 1.0 / n_synth
weights   = [w_real if l == 0 else w_synth for l in train_labels]
sampler   = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

print(f"Class weights — real: {w_real:.5f}  synthetic: {w_synth:.5f}")
print(f"(Sampler ensures balanced batches despite 1:{n_synth//n_real} imbalance)\n")

train_ds = SpectrogramDataset(train_paths, train_labels, train_tf)
val_ds   = SpectrogramDataset(val_paths,   val_labels,   val_tf)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

# ── Model ─────────────────────────────────────────────────────────────────────
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

# ── Training loop ─────────────────────────────────────────────────────────────
print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}")
print("-" * 50)

best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = train_correct = train_total = 0
    for imgs, labels in train_dl:
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        train_loss    += loss.item() * imgs.size(0)
        train_correct += (out.argmax(1) == labels).sum().item()
        train_total   += imgs.size(0)
    scheduler.step()

    model.eval()
    val_loss = val_correct = val_total = 0
    with torch.no_grad():
        for imgs, labels in val_dl:
            out   = model(imgs)
            loss  = criterion(out, labels)
            val_loss    += loss.item() * imgs.size(0)
            val_correct += (out.argmax(1) == labels).sum().item()
            val_total   += imgs.size(0)

    t_acc = train_correct / train_total * 100
    v_acc = val_correct   / val_total   * 100
    flag  = "  ← best" if v_acc > best_val_acc else ""
    print(f"  {epoch:>3}  {train_loss/train_total:>10.4f}  {t_acc:>8.1f}%"
          f"  {val_loss/val_total:>8.4f}  {v_acc:>6.1f}%{flag}")

    if v_acc > best_val_acc:
        best_val_acc = v_acc
        torch.save(model.state_dict(), MODEL_PATH)

print(f"\nBest val accuracy : {best_val_acc:.1f}%")
print(f"Model saved       : {MODEL_PATH}\n")

# ── Detailed val evaluation split by category ────────────────────────────────
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

results = {"genuine": [], "removal": [], "addition": []}

with torch.no_grad():
    for path, label in zip(val_paths, val_labels):
        img  = val_tf(Image.open(path).convert("RGB")).unsqueeze(0)
        out  = model(img)
        prob = torch.softmax(out, dim=1)[0]
        pred = out.argmax(1).item()

        name = Path(path).name
        if name.startswith("genuine_"):          cat = "genuine"
        elif name.startswith("manipulated_"):    cat = "removal"
        else:                                    cat = "addition"

        results[cat].append({
            "correct":        pred == label,
            "conf_synthetic": round(prob[1].item(), 4),
        })

print("=" * 55)
print("  Validation results by category (v2 model)")
print("=" * 55)
for cat, rows in results.items():
    n       = len(rows)
    correct = sum(r["correct"] for r in rows)
    avg_c   = sum(r["conf_synthetic"] for r in rows) / n if n else 0
    print(f"  {cat:<12} : {correct:>3}/{n:<3}  acc={correct/n*100:>5.1f}%"
          f"  avg conf_synth={avg_c:.3f}")
print("=" * 55)

# Save val results to CSV
out_csv = BASE / "results" / "cnn_v2_val_results.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["file", "category", "label", "pred",
                                       "conf_real", "conf_synthetic", "correct"])
    w.writeheader()
    with torch.no_grad():
        for path, label in zip(val_paths, val_labels):
            img  = val_tf(Image.open(path).convert("RGB")).unsqueeze(0)
            out  = model(img)
            prob = torch.softmax(out, dim=1)[0]
            pred = out.argmax(1).item()
            name = Path(path).name
            if name.startswith("genuine_"):       cat = "genuine"
            elif name.startswith("manipulated_"): cat = "removal"
            else:                                 cat = "addition"
            w.writerow({
                "file":           name,
                "category":       cat,
                "label":          "real" if label == 0 else "synthetic",
                "pred":           "real" if pred  == 0 else "synthetic",
                "conf_real":      round(prob[0].item(), 4),
                "conf_synthetic": round(prob[1].item(), 4),
                "correct":        pred == label,
            })

print(f"\nVal CSV saved → {out_csv.name}")
