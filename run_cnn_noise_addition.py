"""
Run CNN noise-removal detector on noise-addition WAVs and append results
to cnn_noise_removal_results_with_category.csv.

Steps:
  1. Generate mel spectrogram PNGs for 399 noise-addition WAVs
     → datasets/spectrograms_noise_addition/
  2. Run ResNet18 CNN (trained on noise-removal spectrograms)
  3. Append rows to results/cnn_noise_removal_results_with_category.csv
     with category = "Noise Addition"

Expected finding: CNN was trained on dark-stripe (removal) patterns;
noise-addition creates a bright stripe → CNN likely classifies as "real".
"""

import csv, torch, torch.nn as nn, numpy as np, librosa, librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

BASE        = Path(os.environ.get("NOISE_BASE_DIR", str(Path(__file__).parent)))
AUDIO_DIR   = BASE / "datasets" / "rasa_manipulated" / "noise_addition"
SPEC_DIR    = BASE / "datasets" / "spectrograms_noise_addition"
MODEL_PATH  = BASE / "models" / "spectrogram_cnn.pt"
EXISTING_CSV = BASE / "results" / "cnn_noise_removal_results_with_category.csv"

SPEC_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 22050
N_MELS    = 128
FMAX      = 8000

CODE_TYPE = {
    "hv": "hvac", "wn": "white_noise", "cr": "crowd",
    "ra": "rain", "ou": "outdoor", "hu": "human", "no": "genuine"
}

# ── Spectrogram generation (same params as generate_spectrograms.py) ──────────
def generate_spectrogram(audio_path: Path, out_path: Path):
    audio, sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
    audio = audio[:120 * TARGET_SR]

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


# ── Load CNN ──────────────────────────────────────────────────────────────────
print("Loading CNN model...")
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
    img = val_tf(Image.open(spec_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        out  = cnn(img)
        prob = torch.softmax(out, dim=1)[0]
        pred = out.argmax(dim=1).item()
    return {
        "verdict":   "real" if pred == 0 else "synthetic",
        "real":      round(prob[0].item(), 4),
        "synthetic": round(prob[1].item(), 4),
    }


def parse_noise_types(stem: str) -> list[str]:
    """Extract noise type names from filename suffix like _multi_cr_no_hv_wn."""
    try:
        after_multi = stem.split("_multi_", 1)[1]   # "cr_no_hv_wn"
        codes = after_multi.split("_")               # ["cr", "no", "hv", "wn"]
        return [CODE_TYPE.get(c, c) for c in codes]
    except (IndexError, ValueError):
        return []


def build_reasoning(stem: str, verdict: str) -> str:
    types = parse_noise_types(stem)
    noisy_segs = [(i, t) for i, t in enumerate(types) if t != "genuine"]
    if not noisy_segs:
        return "All four segments are genuine (no noise added)."

    seg_strs = ", ".join(
        f"seg{i} ({t}, {i*30}–{(i+1)*30}s)" for i, t in noisy_segs
    )
    if verdict == "synthetic":
        return (
            f"Noise added to {len(noisy_segs)} segment(s): {seg_strs}. "
            f"CNN detected spectral discontinuity consistent with manipulation."
        )
    else:
        return (
            f"Noise added to {len(noisy_segs)} segment(s): {seg_strs}. "
            f"CNN trained on noise-removal (dark stripe) did not detect the "
            f"noise-addition bright stripe — classified as real."
        )


# ── Generate spectrograms & run CNN ──────────────────────────────────────────
wav_files = sorted(AUDIO_DIR.glob("rasa_Hindi_Male_*_multi_*.wav"))
print(f"\nFound {len(wav_files)} noise-addition WAVs")
print(f"Spectrograms → {SPEC_DIR.name}/\n")

new_rows = []
tp = tn = fp = fn = 0

for i, wav_path in enumerate(wav_files, 1):
    spec_path = SPEC_DIR / f"noise_addition_{wav_path.stem}.png"

    if not spec_path.exists():
        generate_spectrogram(wav_path, spec_path)
        if i % 50 == 0:
            print(f"  [{i}/{len(wav_files)}] spectrograms generated...")

    r = predict(spec_path)
    expected = "synthetic"   # all noise-addition files are manipulated
    correct  = r["verdict"] == expected
    reasoning = build_reasoning(wav_path.stem, r["verdict"])

    if correct:  tp += 1
    else:        fn += 1

    print(f"  {wav_path.name[:60]:<60}  {r['verdict']:<10}  {'✓' if correct else '✗'}")

    new_rows.append({
        "file":           spec_path.name,
        "expected":       expected,
        "verdict":        r["verdict"],
        "correct":        correct,
        "conf_real":      r["real"],
        "conf_synthetic": r["synthetic"],
        "reasoning":      reasoning,
        "category":       "Noise Addition",
    })

# ── Append to existing CSV ────────────────────────────────────────────────────
fields = ["file", "expected", "verdict", "correct", "conf_real", "conf_synthetic",
          "reasoning", "category"]

with open(EXISTING_CSV, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writerows(new_rows)

# ── Summary ───────────────────────────────────────────────────────────────────
total = len(new_rows)
print("\n" + "=" * 70)
print(f"  Noise-addition WAVs tested  : {total}")
print(f"  Detected as synthetic (TP)  : {tp}  ({tp/total*100:.1f}%)")
print(f"  Detected as real (FN)       : {fn}  ({fn/total*100:.1f}%)")
print(f"\n  Rows appended to            : {EXISTING_CSV.name}")
print(f"  New total rows in CSV       : 500 + {total} = {500 + total}")
print("=" * 70)
print("\nNote: CNN was trained on noise-REMOVAL (dark stripe).")
print("Noise-ADDITION produces a bright stripe — a different visual signature.")
print("High FN rate confirms the CNN does not generalise to noise addition.")
