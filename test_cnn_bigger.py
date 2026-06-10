"""
Bigger CNN test — generate all 4 removal types for all 99 genuine clips.
Skips WAVs and spectrograms that already exist.
Runs CNN only on newly generated (unseen) spectrograms.

Steps:
  1. Generate missing removal WAVs (all 4 types × 99 clips = 396 total)
  2. Generate spectrograms for new WAVs
  3. Run CNN on new spectrograms and report accuracy
"""

import torch, torch.nn as nn, random, numpy as np
import soundfile as sf, librosa, librosa.display, noisereduce as nr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = Path(os.environ.get("NOISE_BASE_DIR", str(Path(__file__).parent)))
GENUINE_DIR = BASE / "datasets" / "rasa"
MANIP_DIR   = BASE / "datasets" / "rasa_manipulated" / "noise_removal"
SPEC_DIR    = BASE / "datasets" / "spectrograms"
MODEL_PATH  = BASE / "models" / "spectrogram_cnn.pt"
MANIP_DIR.mkdir(parents=True, exist_ok=True)
SPEC_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR   = 22050
TOTAL_DUR   = 120
DENOISE_DUR = 30
N_MELS      = 128
FMAX        = 8000
REMOVAL_TYPES = ["start", "mid", "end", "random"]

random.seed(42)

# ── Step 1 — Generate missing removal WAVs ────────────────────────────────────
print("=" * 60)
print("STEP 1 — Generating missing removal WAVs")
print("=" * 60)

all_clips = sorted(GENUINE_DIR.glob("rasa_Hindi_Male_*.wav"))
print(f"Genuine clips : {len(all_clips)}")
print(f"Target        : {len(all_clips) * 4} manipulated WAVs (4 types × {len(all_clips)} clips)\n")

new_wavs = []
skipped  = 0

for clip_path in all_clips:
    audio, sr = librosa.load(str(clip_path), sr=TARGET_SR, mono=True)
    if len(audio) < TOTAL_DUR * TARGET_SR:
        print(f"  SKIP {clip_path.name} — too short")
        continue
    audio = audio[:TOTAL_DUR * TARGET_SR]

    for rtype in REMOVAL_TYPES:
        out_name = f"{clip_path.stem}_removal_{rtype}.wav"
        out_path = MANIP_DIR / out_name

        if out_path.exists():
            skipped += 1
            continue

        # Determine 30s window
        if rtype == "start":
            d_start = 0
        elif rtype == "mid":
            d_start = (TOTAL_DUR // 2) - (DENOISE_DUR // 2)
        elif rtype == "end":
            d_start = TOTAL_DUR - DENOISE_DUR
        else:
            d_start = random.randint(0, TOTAL_DUR - DENOISE_DUR)

        d_end   = d_start + DENOISE_DUR
        s_start = d_start * TARGET_SR
        s_end   = d_end   * TARGET_SR

        manipulated = audio.copy()
        segment     = manipulated[s_start:s_end]
        denoised    = nr.reduce_noise(y=segment, sr=TARGET_SR, stationary=False)
        manipulated[s_start:s_end] = denoised

        sf.write(str(out_path), manipulated, TARGET_SR, subtype="PCM_16")
        new_wavs.append(out_path)
        print(f"  Created: {out_name}  ({rtype}, {d_start}–{d_end}s)")

print(f"\nNew WAVs created : {len(new_wavs)}")
print(f"Already existed  : {skipped}")
print(f"Total now        : {len(list(MANIP_DIR.glob('*.wav')))}")


# ── Step 2 — Generate spectrograms for new WAVs ───────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Generating spectrograms for new WAVs")
print("=" * 60)

def generate_spectrogram(audio_path: Path, out_path: Path):
    audio, sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
    audio     = audio[:TOTAL_DUR * TARGET_SR]

    S    = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, fmax=FMAX)
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), dpi=120,
                                    gridspec_kw={"height_ratios": [3, 1]})

    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel",
                             fmax=FMAX, ax=ax1, cmap="magma")
    for t in [30, 60, 90]:
        ax1.axvline(x=t, color="cyan", linewidth=1.0, linestyle="--", alpha=0.8)
        ax1.text(t + 0.5, 7000, f"{t}s", color="cyan", fontsize=7)
    ax1.set_xlabel("")
    ax1.set_ylabel("Frequency (mel)")

    noise_floor_db = S_db[:20, :]
    vmin = np.percentile(noise_floor_db, 5)
    vmax = np.percentile(noise_floor_db, 95)
    librosa.display.specshow(noise_floor_db, sr=sr, x_axis="time",
                             ax=ax2, cmap="inferno", vmin=vmin, vmax=vmax)
    for t in [30, 60, 90]:
        ax2.axvline(x=t, color="cyan", linewidth=1.0, linestyle="--", alpha=0.8)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Low freq")

    plt.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close()

new_specs = []
for wav_path in new_wavs:
    spec_path = SPEC_DIR / f"manipulated_{wav_path.stem}.png"
    if spec_path.exists():
        continue
    generate_spectrogram(wav_path, spec_path)
    new_specs.append(spec_path)
    print(f"  Saved: {spec_path.name}")

print(f"\nNew spectrograms : {len(new_specs)}")


# ── Step 3 — Run CNN on new spectrograms ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — CNN inference on new unseen spectrograms")
print("=" * 60)

cnn = models.resnet18()
cnn.fc = nn.Linear(cnn.fc.in_features, 2)
cnn.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
cnn.eval()

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict(spec_path: Path) -> dict:
    img  = val_tf(Image.open(spec_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        out  = cnn(img)
        prob = torch.softmax(out, dim=1)[0]
        pred = out.argmax(dim=1).item()
    verdict = "real" if pred == 0 else "synthetic"
    return {
        "verdict":   verdict,
        "real":      round(prob[0].item(), 4),
        "synthetic": round(prob[1].item(), 4),
    }

# Test on all new manipulated spectrograms (label = synthetic)
print(f"\n{'File':<55}  {'Pred':<10}  {'real':>6}  {'synth':>6}  Correct")
print("-" * 90)

tp = fp = tn = fn = 0

for spec in sorted(new_specs):
    r      = predict(spec)
    is_manip = spec.name.startswith("manipulated")
    expected = "synthetic" if is_manip else "real"
    correct  = r["verdict"] == expected

    mark = "✓" if correct else "✗"
    print(f"  {spec.name:<53}  {r['verdict']:<10}  {r['real']:>6.3f}  {r['synthetic']:>6.3f}  {mark}")

    if is_manip and correct:     tp += 1
    elif is_manip and not correct: fn += 1
    elif not is_manip and correct: tn += 1
    else:                          fp += 1

total = tp + tn + fp + fn
print("\n" + "=" * 60)
print(f"  New unseen spectrograms tested : {total}")
print(f"  Correctly detected synthetic   : {tp}/{tp+fn}")
print(f"  Correctly detected real        : {tn}/{tn+fp}")
print(f"  Overall accuracy               : {(tp+tn)/total*100:.1f}%  ({tp+tn}/{total})")
print("=" * 60)
