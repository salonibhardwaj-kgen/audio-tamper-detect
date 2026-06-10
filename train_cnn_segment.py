"""
Train a dedicated segment-level CNN classifier.

Input : 30s segment spectrograms from spectrograms_noise_addition_segs/
Label : genuine segment → 0 (real),  noisy segment → 1 (synthetic)

Class balance: ~794 genuine vs ~802 noisy — nearly perfect, no sampler needed.
Output: models/spectrogram_cnn_seg.pt
"""

import torch, torch.nn as nn, random, csv
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from collections import defaultdict

BASE       = Path(os.environ.get("NOISE_BASE_DIR", str(Path(__file__).parent)))
SEG_DIR    = BASE / "datasets" / "spectrograms_noise_addition_segs"
MODEL_DIR  = BASE / "models"
MODEL_PATH = MODEL_DIR / "spectrogram_cnn_seg.pt"
MODEL_DIR.mkdir(exist_ok=True)

EPOCHS     = 20
BATCH_SIZE = 8
LR         = 1e-4
SEED       = 42
random.seed(SEED)
torch.manual_seed(SEED)

CODE_TYPE = {
    "hv": "hvac", "wn": "white_noise", "cr": "crowd",
    "ra": "rain",  "ou": "outdoor",    "hu": "human", "no": "genuine"
}

def parse_label(spec_path: Path) -> tuple[int, str]:
    """Returns (label, actual_noise_type) from segment filename."""
    stem     = spec_path.stem                        # e.g. rasa_..._multi_cr_no_hv_no_seg2
    seg_idx  = int(stem[-1])
    wav_stem = stem[:-5]                             # strip _seg2
    after    = wav_stem.split("_multi_", 1)[1]       # e.g. cr_no_hv_no
    codes    = after.split("_")[:4]
    ntype    = CODE_TYPE.get(codes[seg_idx], "genuine")
    label    = 0 if ntype == "genuine" else 1
    return label, ntype

# ── Load all segment PNGs ─────────────────────────────────────────────────────
all_specs = sorted(SEG_DIR.glob("*_seg?.png"))
all_paths, all_labels, all_types = [], [], []

type_count: dict[str, int] = defaultdict(int)
for p in all_specs:
    label, ntype = parse_label(p)
    all_paths.append(p)
    all_labels.append(label)
    all_types.append(ntype)
    type_count[ntype] += 1

print("Dataset breakdown:")
for t, n in sorted(type_count.items()):
    lbl = "real=0" if t == "genuine" else "synth=1"
    print(f"  {t:<14} ({lbl}): {n}")
print(f"  {'─'*35}")
print(f"  Total segments : {len(all_paths)}")
print(f"  Real  (label 0): {all_labels.count(0)}")
print(f"  Synth (label 1): {all_labels.count(1)}\n")

# ── Train / val split (80/20) ─────────────────────────────────────────────────
combined = list(zip(all_paths, all_labels, all_types))
random.shuffle(combined)
all_paths, all_labels, all_types = zip(*combined)

split        = int(0.8 * len(all_paths))
train_paths  = list(all_paths[:split]);  train_labels = list(all_labels[:split])
val_paths    = list(all_paths[split:]);  val_labels   = list(all_labels[split:])
val_types    = list(all_types[split:])

print(f"Train : {len(train_paths)} segments")
print(f"Val   : {len(val_paths)} segments\n")

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

class SegDataset(Dataset):
    def __init__(self, paths, labels, tf):
        self.paths = paths; self.labels = labels; self.tf = tf
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), self.labels[i]

train_dl = DataLoader(SegDataset(train_paths, train_labels, train_tf),
                      batch_size=BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(SegDataset(val_paths, val_labels, val_tf),
                      batch_size=BATCH_SIZE)

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
    t_loss = t_correct = t_total = 0
    for imgs, labels in train_dl:
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward(); optimizer.step()
        t_loss    += loss.item() * imgs.size(0)
        t_correct += (out.argmax(1) == labels).sum().item()
        t_total   += imgs.size(0)
    scheduler.step()

    model.eval()
    v_loss = v_correct = v_total = 0
    with torch.no_grad():
        for imgs, labels in val_dl:
            out    = model(imgs)
            loss   = criterion(out, labels)
            v_loss    += loss.item() * imgs.size(0)
            v_correct += (out.argmax(1) == labels).sum().item()
            v_total   += imgs.size(0)

    t_acc = t_correct / t_total * 100
    v_acc = v_correct / v_total * 100
    flag  = "  ← best" if v_acc > best_val_acc else ""
    print(f"  {epoch:>3}  {t_loss/t_total:>10.4f}  {t_acc:>8.1f}%"
          f"  {v_loss/v_total:>8.4f}  {v_acc:>6.1f}%{flag}")
    if v_acc > best_val_acc:
        best_val_acc = v_acc
        torch.save(model.state_dict(), MODEL_PATH)

print(f"\nBest val accuracy : {best_val_acc:.1f}%")
print(f"Model saved       : {MODEL_PATH}\n")

# ── Per noise-type breakdown on val set ───────────────────────────────────────
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

type_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0, "conf_sum": 0.0})
rows = []

with torch.no_grad():
    for path, label, ntype in zip(val_paths, val_labels, val_types):
        img  = val_tf(Image.open(path).convert("RGB")).unsqueeze(0)
        out  = model(img)
        prob = torch.softmax(out, dim=1)[0]
        pred = out.argmax(1).item()
        correct = pred == label

        type_stats[ntype]["total"]    += 1
        type_stats[ntype]["correct"]  += int(correct)
        type_stats[ntype]["conf_sum"] += prob[1].item()

        rows.append({
            "file":           Path(path).name,
            "actual_type":    ntype,
            "label":          "real" if label == 0 else "synthetic",
            "pred":           "real" if pred  == 0 else "synthetic",
            "conf_synthetic": round(prob[1].item(), 4),
            "conf_real":      round(prob[0].item(), 4),
            "correct":        correct,
        })

print(f"{'Noise Type':<14}  {'Val Segs':>8}  {'Correct':>7}  {'Rate':>6}  {'Avg Conf Synth':>14}")
print("-" * 60)
for t, s in sorted(type_stats.items()):
    n   = s["total"]
    c   = s["correct"]
    avg = s["conf_sum"] / n
    print(f"  {t:<12}  {n:>8}  {c:>7}  {c/n*100:>5.1f}%  {avg:>14.3f}")

# Save val CSV
out_csv = BASE / "results" / "cnn_seg_val_results.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["file","actual_type","label","pred",
                                       "conf_synthetic","conf_real","correct"])
    w.writeheader(); w.writerows(rows)
print(f"\nVal CSV → {out_csv.name}")
