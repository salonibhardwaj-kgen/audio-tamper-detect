"""
Segment-wise CNN analysis of noise-addition clips.

For each of the 399 noise-addition WAVs:
  - Split into 4 × 30s segments
  - Generate a spectrogram PNG for each segment
  - Run ResNet18 CNN → real/synthetic confidence per segment
  - Identify which segment has the highest synthetic confidence
    and what noise type it actually is (from the filename ground truth)

Output:
  results/cnn_noise_addition_segmentwise.csv
  One row per clip, with per-segment verdicts + "dominant detection" columns.
"""

import csv, torch, torch.nn as nn, numpy as np, librosa, librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

BASE       = Path(os.environ.get("NOISE_BASE_DIR", str(Path(__file__).parent)))
AUDIO_DIR  = BASE / "datasets" / "rasa_manipulated" / "noise_addition"
SEG_DIR    = BASE / "datasets" / "spectrograms_noise_addition_segs"
MODEL_PATH = BASE / "models" / "spectrogram_cnn.pt"
OUT_CSV    = BASE / "results" / "cnn_noise_addition_segmentwise.csv"

SEG_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 22050
SEG_DUR   = 30
N_MELS    = 128
FMAX      = 8000

CODE_TYPE = {
    "hv": "hvac", "wn": "white_noise", "cr": "crowd",
    "ra": "rain",  "ou": "outdoor",    "hu": "human", "no": "genuine"
}

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
        "conf_real":      round(prob[0].item(), 4),
        "conf_synthetic": round(prob[1].item(), 4),
    }

# ── Spectrogram for a single 30s segment ─────────────────────────────────────
def generate_segment_spectrogram(audio: np.ndarray, sr: int, out_path: Path,
                                  seg_idx: int):
    S    = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, fmax=FMAX)
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), dpi=120,
                                    gridspec_kw={"height_ratios": [3, 1]})

    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel",
                             fmax=FMAX, ax=ax1, cmap="magma")
    t_offset = seg_idx * SEG_DUR
    ax1.set_title(f"Segment {seg_idx}  ({t_offset}–{t_offset+SEG_DUR}s)", fontsize=9)
    ax1.set_xlabel("")
    ax1.set_ylabel("Frequency (mel)")

    noise_floor_db = S_db[:20, :]
    vmin = np.percentile(noise_floor_db, 5)
    vmax = np.percentile(noise_floor_db, 95)
    librosa.display.specshow(noise_floor_db, sr=sr, x_axis="time",
                             ax=ax2, cmap="inferno", vmin=vmin, vmax=vmax)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Low freq")

    plt.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close()


def parse_seg_types(stem: str) -> list[str]:
    """Returns list of 4 noise-type names from filename stem."""
    try:
        after_multi = stem.split("_multi_", 1)[1]   # e.g. "cr_no_hv_wn"
        codes = after_multi.split("_")               # ["cr", "no", "hv", "wn"]
        return [CODE_TYPE.get(c, c) for c in codes[:4]]
    except (IndexError, ValueError):
        return ["unknown"] * 4


# ── Process all clips ─────────────────────────────────────────────────────────
wav_files = sorted(AUDIO_DIR.glob("rasa_Hindi_Male_*_multi_*.wav"))
print(f"Found {len(wav_files)} noise-addition clips\n")

rows = []

for clip_idx, wav_path in enumerate(wav_files, 1):
    seg_types = parse_seg_types(wav_path.stem)   # ground-truth type per segment
    audio, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
    audio = audio[:120 * TARGET_SR]

    seg_results = []   # list of dicts, one per segment

    for seg_i in range(4):
        start  = seg_i * SEG_DUR * TARGET_SR
        end    = start + SEG_DUR * TARGET_SR
        seg_audio = audio[start:end]

        spec_name = f"{wav_path.stem}_seg{seg_i}.png"
        spec_path = SEG_DIR / spec_name

        if not spec_path.exists():
            generate_segment_spectrogram(seg_audio, TARGET_SR, spec_path, seg_i)

        r = predict(spec_path)
        seg_results.append({
            "actual_type":    seg_types[seg_i],
            "verdict":        r["verdict"],
            "conf_real":      r["conf_real"],
            "conf_synthetic": r["conf_synthetic"],
        })

    # ── Dominant detection: segment with highest synthetic confidence ──────────
    best_seg  = max(range(4), key=lambda i: seg_results[i]["conf_synthetic"])
    best_res  = seg_results[best_seg]

    # Among actually-noisy segments only (exclude genuine segments)
    noisy_idxs = [i for i in range(4) if seg_types[i] != "genuine"]
    if noisy_idxs:
        best_noisy_seg = max(noisy_idxs,
                             key=lambda i: seg_results[i]["conf_synthetic"])
        best_noisy_res = seg_results[best_noisy_seg]
    else:
        best_noisy_seg = None
        best_noisy_res = {"actual_type": "—", "conf_synthetic": 0.0, "verdict": "—"}

    row = {
        "clip": wav_path.name,
        # Per-segment columns
        "seg0_actual_type":    seg_results[0]["actual_type"],
        "seg0_verdict":        seg_results[0]["verdict"],
        "seg0_conf_real":      seg_results[0]["conf_real"],
        "seg0_conf_synthetic": seg_results[0]["conf_synthetic"],

        "seg1_actual_type":    seg_results[1]["actual_type"],
        "seg1_verdict":        seg_results[1]["verdict"],
        "seg1_conf_real":      seg_results[1]["conf_real"],
        "seg1_conf_synthetic": seg_results[1]["conf_synthetic"],

        "seg2_actual_type":    seg_results[2]["actual_type"],
        "seg2_verdict":        seg_results[2]["verdict"],
        "seg2_conf_real":      seg_results[2]["conf_real"],
        "seg2_conf_synthetic": seg_results[2]["conf_synthetic"],

        "seg3_actual_type":    seg_results[3]["actual_type"],
        "seg3_verdict":        seg_results[3]["verdict"],
        "seg3_conf_real":      seg_results[3]["conf_real"],
        "seg3_conf_synthetic": seg_results[3]["conf_synthetic"],

        # Dominant detection across ALL 4 segments
        "highest_conf_seg":        best_seg,
        "highest_conf_type":       best_res["actual_type"],
        "highest_conf_synthetic":  best_res["conf_synthetic"],
        "highest_conf_verdict":    best_res["verdict"],

        # Dominant detection among NOISY segments only
        "best_noisy_seg":          best_noisy_seg if best_noisy_seg is not None else "—",
        "best_noisy_type":         best_noisy_res["actual_type"],
        "best_noisy_conf":         best_noisy_res["conf_synthetic"],
        "best_noisy_verdict":      best_noisy_res["verdict"],
    }
    rows.append(row)

    # Progress print
    noisy_str = " | ".join(
        f"seg{i}({seg_results[i]['actual_type'][:3]})={seg_results[i]['conf_synthetic']:.3f}"
        for i in range(4)
    )
    print(f"[{clip_idx:>3}/{len(wav_files)}] {wav_path.name[:50]:<50}  {noisy_str}")
    print(f"         → best noisy: seg{best_noisy_seg} ({best_noisy_res['actual_type']}) "
          f"conf={best_noisy_res['conf_synthetic']:.3f}  verdict={best_noisy_res['verdict']}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
fields = [
    "clip",
    "seg0_actual_type", "seg0_verdict", "seg0_conf_real", "seg0_conf_synthetic",
    "seg1_actual_type", "seg1_verdict", "seg1_conf_real", "seg1_conf_synthetic",
    "seg2_actual_type", "seg2_verdict", "seg2_conf_real", "seg2_conf_synthetic",
    "seg3_actual_type", "seg3_verdict", "seg3_conf_real", "seg3_conf_synthetic",
    "highest_conf_seg", "highest_conf_type", "highest_conf_synthetic", "highest_conf_verdict",
    "best_noisy_seg",  "best_noisy_type",  "best_noisy_conf",  "best_noisy_verdict",
]

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

# ── Summary stats ─────────────────────────────────────────────────────────────
total = len(rows)
detected_clips = sum(
    1 for r in rows
    if any(r[f"seg{i}_verdict"] == "synthetic" for i in range(4))
)
best_noisy_correct = sum(
    1 for r in rows if r["best_noisy_verdict"] == "synthetic"
)

print(f"\n{'='*70}")
print(f"  Total clips analysed             : {total}")
print(f"  Clips with ≥1 segment detected   : {detected_clips}  ({detected_clips/total*100:.1f}%)")
print(f"  Best noisy seg detected synthetic : {best_noisy_correct}  ({best_noisy_correct/total*100:.1f}%)")
print(f"\n  CSV saved → {OUT_CSV.name}")
print(f"{'='*70}")

# Per noise-type breakdown
print("\n── Per noise-type synthetic detection rate (segment level) ──")
type_counts: dict[str, list] = {}
for r in rows:
    for i in range(4):
        t = r[f"seg{i}_actual_type"]
        v = r[f"seg{i}_verdict"]
        if t not in type_counts:
            type_counts[t] = []
        type_counts[t].append(v == "synthetic")

for t, hits in sorted(type_counts.items()):
    n = len(hits)
    s = sum(hits)
    print(f"  {t:<12} : {s:>3}/{n:<3}  ({s/n*100:>5.1f}%)")
