"""
Retrain segment CNN with ALL manipulation types:
  - Existing: noise addition segments (genuine + synthetic from spectrograms_noise_addition_segs/)
  - New:      noisereduce removal segments (extract manipulated 30s from WAVs)
  - New:      audacity removal segments    (extract manipulated 30s from WAVs)

Final training set:
  Genuine  segments: ~794  (existing, from noise addition dataset)
  Synthetic segments: 802  (noise addition) + 300 noisereduce + 300 audacity = 1402
  Total: ~2196 segments

Saved model: models/spectrogram_cnn_seg_v2.pt
"""

import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import random

BASE       = Path(os.environ.get("NOISE_BASE_DIR", str(Path(__file__).parent)))
SR         = 22050
N_MELS     = 128
FMAX       = 8000
N_FFT      = 2048
HOP        = 512

# Existing noise addition segments (all synthetic — no genuine in this dir)
ADDITION_SEG_DIR = BASE / "datasets" / "spectrograms_noise_addition_segs"

# Genuine WAV source
GENUINE_WAV = BASE / "datasets" / "rasa"

# WAV sources for removal
NOISEREDUCE_WAV  = BASE / "datasets" / "rasa_manipulated" / "noise_removal"
AUDACITY_WAV     = BASE / "datasets" / "rasa_manipulated" / "audacity_removal"

# Output dirs
GENUINE_SEG_DIR     = BASE / "datasets" / "spectrograms_genuine_segs"
NR_SEG_DIR          = BASE / "datasets" / "spectrograms_noisereduce_segs"
AUD_SEG_DIR         = BASE / "datasets" / "spectrograms_audacity_segs"
PARTIAL_NOISE_DIR   = BASE / "datasets" / "spectrograms_partial_noise_segs"

GENUINE_SEG_DIR.mkdir(exist_ok=True)
NR_SEG_DIR.mkdir(exist_ok=True)
AUD_SEG_DIR.mkdir(exist_ok=True)

MODEL_OUT = BASE / "models" / "spectrogram_cnn_seg_v3.pt"
VAL_CSV   = BASE / "results" / "cnn_seg_v3_val_results.csv"

# Segment time ranges for each variant
VARIANT_SEGMENTS = {
    "start":  (0,   30),
    "mid":    (45,  75),
    "end":    (90, 120),
}

# ── Generate removal segment spectrogram ───────────────────────────────────
def make_seg_spec(audio, sr, seg_start, seg_end, out_path):
    start_s = int(seg_start * sr)
    end_s   = int(seg_end   * sr)
    segment = audio[start_s:end_s]
    if len(segment) < (seg_end - seg_start) * sr:
        segment = np.pad(segment, (0, (seg_end - seg_start) * sr - len(segment)))

    S    = librosa.feature.melspectrogram(y=segment, sr=sr,
                                          n_mels=N_MELS, fmax=FMAX)
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, axes = plt.subplots(2, 1, figsize=(6, 4),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("black")
    axes[0].imshow(S_db, aspect="auto", origin="lower",
                   cmap="magma", vmin=-80, vmax=0)
    axes[0].axis("off")
    floor = S_db[:20, :]
    axes[1].imshow(floor, aspect="auto", origin="lower", cmap="inferno",
                   vmin=np.percentile(floor, 5),
                   vmax=np.percentile(floor, 95))
    axes[1].axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(str(out_path), dpi=100, bbox_inches="tight", facecolor="black")
    plt.close()


# ── Generate genuine segment spectrograms ─────────────────────────────────
def generate_genuine_segs(wav_dir, out_dir):
    generated = 0
    for wav_path in sorted(wav_dir.glob("*.wav")):
        for variant, (seg_start, seg_end) in VARIANT_SEGMENTS.items():
            out_path = out_dir / f"genuine_{wav_path.stem}_{variant}.png"
            if out_path.exists():
                generated += 1
                continue
            audio, sr = librosa.load(str(wav_path), sr=SR, mono=True)
            make_seg_spec(audio, sr, seg_start, seg_end, out_path)
            generated += 1
    print(f"  genuine: {generated} segment spectrograms ready in {out_dir.name}/")
    return generated


# ── Generate spectrograms for removal WAVs (start/mid/end only) ────────────
def generate_removal_segs(wav_dir, out_dir, tool_tag):
    generated = 0
    for wav_path in sorted(wav_dir.glob("*.wav")):
        name = wav_path.stem
        variant = None
        for v in VARIANT_SEGMENTS:
            if f"_{v}" in name:
                variant = v
                break
        if variant is None:
            continue   # skip 'random' — offset unknown

        out_path = out_dir / f"{tool_tag}_{name}.png"
        if out_path.exists():
            generated += 1
            continue

        audio, sr = librosa.load(str(wav_path), sr=SR, mono=True)
        seg_start, seg_end = VARIANT_SEGMENTS[variant]
        make_seg_spec(audio, sr, seg_start, seg_end, out_path)
        generated += 1

    print(f"  {tool_tag}: {generated} segment spectrograms ready in {out_dir.name}/")
    return generated


# ── Dataset ────────────────────────────────────────────────────────────────
class SegDataset(Dataset):
    def __init__(self, items, transform):
        self.items     = items     # list of (path, label)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Generate segment spectrograms
    print("Generating segment spectrograms...")
    generate_genuine_segs(GENUINE_WAV,    GENUINE_SEG_DIR)
    generate_removal_segs(NOISEREDUCE_WAV, NR_SEG_DIR,  "nr")
    generate_removal_segs(AUDACITY_WAV,   AUD_SEG_DIR, "aud")

    # 2. Collect all items with labels
    items = []

    # Genuine segments → label 0
    gen_items = [(p, 0) for p in sorted(GENUINE_SEG_DIR.glob("genuine_*.png"))]
    items.extend(gen_items)
    print(f"\nGenuine segments        : {len(gen_items)}")

    # Noise addition segments → all synthetic (label 1)
    add_before = len(items)
    for p in sorted(ADDITION_SEG_DIR.glob("*.png")):
        items.append((p, 1))
    print(f"Noise addition segments : {len(items) - add_before}")

    # Noisereduce removal segments → label 1
    nr_items = [(p, 1) for p in sorted(NR_SEG_DIR.glob("nr_*.png"))]
    items.extend(nr_items)
    print(f"Noisereduce removal segs: {len(nr_items)}")

    # Audacity removal segments → label 1
    aud_items = [(p, 1) for p in sorted(AUD_SEG_DIR.glob("aud_*.png"))]
    items.extend(aud_items)
    print(f"Audacity removal segs   : {len(aud_items)}")

    # Partial noise addition segments → label 1
    partial_items = [(p, 1) for p in sorted(PARTIAL_NOISE_DIR.glob("partial_*.png"))]
    items.extend(partial_items)
    print(f"Partial noise segs      : {len(partial_items)}")

    # Assamese noise addition segments (full + partial) → label 1
    AS_ADD_DIR = BASE / "datasets" / "spectrograms_assamese_noise_addition_segs"
    as_items   = [(p, 1) for p in sorted(AS_ADD_DIR.glob("as_*.png"))]
    items.extend(as_items)
    print(f"Assamese noise add segs : {len(as_items)}")

    # Count
    genuine_n  = sum(1 for _, l in items if l == 0)
    synthetic_n = sum(1 for _, l in items if l == 1)
    print(f"\nTotal  : {len(items)}")
    print(f"Genuine (0): {genuine_n}  |  Synthetic (1): {synthetic_n}")

    # 3. Train / val split
    random.shuffle(items)
    split      = int(0.8 * len(items))
    train_items = items[:split]
    val_items   = items[split:]
    print(f"Train  : {len(train_items)}  |  Val: {len(val_items)}")

    # 4. Transforms
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # 5. Weighted sampler (handle imbalance)
    train_labels = [l for _, l in train_items]
    class_counts = [train_labels.count(0), train_labels.count(1)]
    weights      = [1.0 / class_counts[l] for l in train_labels]
    sampler      = WeightedRandomSampler(weights, len(weights))

    train_ds = SegDataset(train_items, train_tf)
    val_ds   = SegDataset(val_items,   val_tf)
    train_dl = DataLoader(train_ds, batch_size=16, sampler=sampler)
    val_dl   = DataLoader(val_ds,   batch_size=16, shuffle=False)

    # 6. Model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nDevice : {device}")

    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)

    optimiser  = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler  = torch.optim.lr_scheduler.StepLR(optimiser, step_size=5, gamma=0.5)
    criterion  = nn.CrossEntropyLoss()

    # 7. Training loop
    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>10}  "
          f"{'Val Loss':>9}  {'Val Acc':>8}")
    print("-" * 55)

    best_val_acc = 0.0
    EPOCHS = 20

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss, t_correct, t_total = 0, 0, 0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimiser.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimiser.step()
            t_loss    += loss.item() * len(labels)
            t_correct += (out.argmax(1) == labels).sum().item()
            t_total   += len(labels)

        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                out  = model(imgs)
                loss = criterion(out, labels)
                v_loss    += loss.item() * len(labels)
                v_correct += (out.argmax(1) == labels).sum().item()
                v_total   += len(labels)

        t_acc = t_correct / t_total
        v_acc = v_correct / v_total
        tag   = " ← best" if v_acc > best_val_acc else ""
        print(f"{epoch:>6}  {t_loss/t_total:>10.4f}  {t_acc*100:>9.1f}%  "
              f"{v_loss/v_total:>9.4f}  {v_acc*100:>7.1f}%{tag}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), str(MODEL_OUT))

        scheduler.step()

    print(f"\nBest val accuracy : {best_val_acc*100:.1f}%")
    print(f"Model saved       : {MODEL_OUT}")

    # 8. Per-category validation breakdown
    print("\nPer-category breakdown on val set:")
    model.load_state_dict(torch.load(str(MODEL_OUT), weights_only=True))
    model.eval()

    categories = {"genuine": [], "noise_addition": [], "partial_noise": [], "noisereduce": [], "audacity": []}
    with torch.no_grad():
        for path, label in val_items:
            img  = val_tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            out  = model(img)
            pred = out.argmax(1).item()
            conf = torch.softmax(out, dim=1)[0][1].item()
            correct = int(pred == label)

            stem = path.stem
            if "genuine" in stem and "nr_" not in stem and "aud_" not in stem and "partial_" not in stem:
                categories["genuine"].append((correct, conf))
            elif "nr_" in stem:
                categories["noisereduce"].append((correct, conf))
            elif "aud_" in stem:
                categories["audacity"].append((correct, conf))
            elif "partial_" in stem:
                categories["partial_noise"].append((correct, conf))
            else:
                categories["noise_addition"].append((correct, conf))

    print(f"\n{'Category':<20} {'Val':<6} {'Correct':<8} {'Rate':<8} {'Avg Conf'}")
    print("-" * 60)
    for cat, results in categories.items():
        if not results:
            continue
        n       = len(results)
        correct = sum(r[0] for r in results)
        avg_c   = sum(r[1] for r in results) / n
        print(f"  {cat:<18} {n:<6} {correct:<8} {correct/n*100:>5.1f}%  "
              f"  {avg_c:.3f}")
