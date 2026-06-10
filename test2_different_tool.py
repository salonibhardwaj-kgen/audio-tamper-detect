"""
TEST 2 — Different noise removal tool (spectral subtraction).
noisereduce uses a Wiener filter. This implements classic spectral subtraction
(Boll 1979) — a completely different algorithm, different artifact shape.
If CNN detects it → learned generic noise-removal artifacts
If CNN misses it  → learned noisereduce's specific fingerprint only
"""

import numpy as np, librosa, librosa.display, scipy.signal
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

BASE       = Path(os.environ.get("NOISE_BASE_DIR", str(Path(__file__).parent)))
RASA_DIR   = BASE / "datasets" / "rasa"
MODEL_PATH = BASE / "models" / "spectrogram_cnn.pt"
OUT_DIR    = BASE / "results" / "test_generalization"
OUT_DIR.mkdir(exist_ok=True)

TARGET_SR = 22050
N_MELS = 128; FMAX = 8000

def spectral_subtraction(audio: np.ndarray, sr: int,
                          noise_frames: int = 20,
                          alpha: float = 2.0) -> np.ndarray:
    """Classic spectral subtraction (Boll 1979).
    Estimates noise from first `noise_frames` STFT frames,
    subtracts alpha × noise power from all frames."""
    n_fft   = 2048
    hop     = 512
    S       = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
    mag     = np.abs(S)
    phase   = np.angle(S)

    # Noise estimate from first few frames (assumed noise-only)
    noise_est = np.mean(mag[:, :noise_frames], axis=1, keepdims=True)

    # Subtract and half-wave rectify (no negative energy)
    mag_clean = np.maximum(mag - alpha * noise_est, 0.01 * mag)

    S_clean = mag_clean * np.exp(1j * phase)
    return librosa.istft(S_clean, hop_length=hop, length=len(audio))

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

# ── Load genuine clip ─────────────────────────────────────────────────────────
clip = sorted(RASA_DIR.glob("*.wav"))[0]
audio, _ = librosa.load(str(clip), sr=TARGET_SR, mono=True)
audio = audio[:120 * TARGET_SR]
print(f"Loaded: {clip.name}")

# ── Apply spectral subtraction to segment 1 (30-60s) ─────────────────────────
seg_samples  = 30 * TARGET_SR
manipulated  = audio.copy()
seg          = manipulated[seg_samples: 2 * seg_samples]
cleaned      = spectral_subtraction(seg, TARGET_SR)
manipulated[seg_samples: 2 * seg_samples] = cleaned
print("Applied spectral subtraction to segment 1 (30–60s)")

# ── Also apply noisereduce for comparison ────────────────────────────────────
import noisereduce as nr
manip_nr    = audio.copy()
seg_nr      = nr.reduce_noise(y=audio[seg_samples: 2*seg_samples], sr=TARGET_SR, prop_decrease=1.0)
manip_nr[seg_samples: 2*seg_samples] = seg_nr

# ── Generate spectrograms ─────────────────────────────────────────────────────
genuine_spec  = OUT_DIR / "test2_genuine.png"
ss_spec       = OUT_DIR / "test2_spectral_subtraction.png"
nr_spec       = OUT_DIR / "test2_noisereduce.png"

make_spec(audio,        genuine_spec, f"Genuine — {clip.name}")
make_spec(manipulated,  ss_spec,      "Spectral subtraction on seg1 (different tool)")
make_spec(manip_nr,     nr_spec,      "noisereduce on seg1 (original tool)")
print("Spectrograms generated")

# ── Run CNN ───────────────────────────────────────────────────────────────────
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

vg, rg, sg  = predict(genuine_spec)
vss, rss, sss = predict(ss_spec)
vnr, rnr, snr = predict(nr_spec)

print("\n" + "="*60)
print("TEST 2 — DIFFERENT NOISE REMOVAL TOOL")
print("="*60)
print(f"  Genuine             → {vg:<10} (real={rg:.3f}, synth={sg:.3f})")
print(f"  Spectral subtraction → {vss:<10} (real={rss:.3f}, synth={sss:.3f})")
print(f"  noisereduce         → {vnr:<10} (real={rnr:.3f}, synth={snr:.3f})")
print()
if vnr == "synthetic" and vss == "synthetic":
    print("  RESULT: ✓ CNN detects BOTH tools → learned generic noise-removal artifact")
elif vnr == "synthetic" and vss != "synthetic":
    print("  RESULT: ✗ CNN only detects noisereduce → learned tool-specific fingerprint")
elif vnr != "synthetic" and vss == "synthetic":
    print("  RESULT: ✗ Unexpected — detects spectral sub but not noisereduce")
print("="*60)
