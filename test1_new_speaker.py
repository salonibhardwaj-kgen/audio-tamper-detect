"""
TEST 1 — New speaker / unseen audio type.
Uses ESC-50 environmental sounds (never in training) — not speech at all.
Concatenates ESC-50 clips → 120s, applies noisereduce to seg1 (30-60s),
generates spectrogram, runs CNN v1.
If CNN detects synthetic → it learned the dark stripe, not speaker identity.
If CNN misses it        → it learned Rasa speaker patterns, not the artifact.
"""

import numpy as np, librosa, librosa.display, noisereduce as nr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn, soundfile as sf
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

BASE       = Path(os.environ.get("NOISE_BASE_DIR", str(Path(__file__).parent)))
ESC_DIR    = BASE / "datasets" / "ESC-50-master" / "audio"
MODEL_PATH = BASE / "models" / "spectrogram_cnn.pt"
OUT_DIR    = BASE / "results" / "test_generalization"
OUT_DIR.mkdir(exist_ok=True)

TARGET_SR = 22050
N_MELS = 128; FMAX = 8000

# ── Build 120s clip from ESC-50 files ────────────────────────────────────────
esc_files = sorted(ESC_DIR.glob("*.wav"))[:40]
segments = []
for f in esc_files:
    a, _ = librosa.load(str(f), sr=TARGET_SR, mono=True)
    segments.append(a)

combined = np.concatenate(segments)
audio_120 = combined[:120 * TARGET_SR]
if len(audio_120) < 120 * TARGET_SR:
    audio_120 = np.tile(combined, 3)[:120 * TARGET_SR]

print(f"Built 120s ESC-50 clip from {len(esc_files)} files")

# ── Apply noisereduce to segment 1 (30-60s) only ─────────────────────────────
seg_samples = 30 * TARGET_SR
manipulated = audio_120.copy()
seg = manipulated[seg_samples: 2 * seg_samples]
cleaned = nr.reduce_noise(y=seg, sr=TARGET_SR, prop_decrease=1.0)
manipulated[seg_samples: 2 * seg_samples] = cleaned
print("Applied noisereduce to segment 1 (30–60s)")

# ── Generate spectrograms for both genuine and manipulated ───────────────────
def make_spec(audio, path, title):
    S    = librosa.feature.melspectrogram(y=audio, sr=TARGET_SR, n_mels=N_MELS, fmax=FMAX)
    S_db = librosa.power_to_db(S, ref=np.max)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), dpi=120,
                                    gridspec_kw={"height_ratios": [3, 1]})
    librosa.display.specshow(S_db, sr=TARGET_SR, x_axis="time", y_axis="mel",
                             fmax=FMAX, ax=ax1, cmap="magma")
    ax1.set_title(title)
    for t in [30, 60, 90]:
        ax1.axvline(x=t, color="cyan", lw=1.0, ls="--", alpha=0.8)
        ax1.text(t+0.5, 7000, f"{t}s", color="cyan", fontsize=7)
    ax1.set_xlabel(""); ax1.set_ylabel("Frequency (mel)")
    nf = S_db[:20, :]
    librosa.display.specshow(nf, sr=TARGET_SR, x_axis="time", ax=ax2,
                             cmap="inferno", vmin=np.percentile(nf,5), vmax=np.percentile(nf,95))
    for t in [30, 60, 90]:
        ax2.axvline(x=t, color="cyan", lw=1.0, ls="--", alpha=0.8)
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Low freq")
    plt.tight_layout(); fig.savefig(str(path), bbox_inches="tight"); plt.close()

genuine_spec    = OUT_DIR / "test1_esc50_genuine.png"
manipulated_spec = OUT_DIR / "test1_esc50_manipulated.png"
make_spec(audio_120,   genuine_spec,    "ESC-50 Genuine (never in training)")
make_spec(manipulated, manipulated_spec, "ESC-50 + noisereduce on seg1 (30-60s)")
print("Spectrograms generated")

# ── Load CNN v1 and predict ───────────────────────────────────────────────────
cnn = models.resnet18()
cnn.fc = nn.Linear(cnn.fc.in_features, 2)
cnn.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
cnn.eval()

val_tf = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def predict(p):
    img = val_tf(Image.open(p).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        out  = cnn(img)
        prob = torch.softmax(out, dim=1)[0]
        pred = out.argmax(1).item()
    return "real" if pred==0 else "synthetic", prob[0].item(), prob[1].item()

v1, r1, s1 = predict(genuine_spec)
v2, r2, s2 = predict(manipulated_spec)

print("\n" + "="*60)
print("TEST 1 — NEW AUDIO TYPE (ESC-50, never in training)")
print("="*60)
print(f"  Genuine ESC-50   → CNN says: {v1:<10} (real={r1:.3f}, synth={s1:.3f})")
print(f"  Manipulated (NR) → CNN says: {v2:<10} (real={r2:.3f}, synth={s2:.3f})")
print()
if v1 == "real" and v2 == "synthetic":
    print("  RESULT: ✓ CNN detected the dark stripe on UNSEEN audio type")
    print("          → Learned the artifact pattern, NOT speaker identity")
elif v1 == "real" and v2 == "real":
    print("  RESULT: ✗ CNN missed the manipulation on unseen audio")
    print("          → Likely learned Rasa-specific patterns, not the stripe")
elif v1 == "synthetic":
    print("  RESULT: ✗ CNN false-positive on genuine ESC-50")
    print("          → Model not generalising correctly")
print("="*60)
