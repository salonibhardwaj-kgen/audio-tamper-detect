"""
AI4Bharat IndicVoices Hindi — genuine recording baseline test.

IndicVoices clips are short utterances (5–60s each). This script
concatenates clips from the SAME speaker until we reach 120s, then
saves that as a single WAV. All audio comes from one volunteer's
recording session, so the noise floor should be stationary.

Collects 50 such speaker-concatenated clips.

WAVs  → datasets/indicvoices_hindi/
CSV   → results/indicvoices_genuine.csv
"""

import sys, csv, os
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit(
        "Set HF_TOKEN before running:\n"
        "  export HF_TOKEN=hf_...\n"
        "Then accept dataset terms at: https://huggingface.co/datasets/ai4bharat/IndicVoices"
    )

from noise_analyzer import analyze_file, IS_THRESHOLD, SFM_THRESHOLD

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_DUR = 120        # seconds per output clip
TARGET_SR  = 22050      # resample target
N_FILES    = 50         # clips to collect

BASE    = Path(os.environ.get("NOISE_BASE_DIR", str(Path(__file__).parent)))
OUT_DIR = BASE / "datasets" / "indicvoices_hindi"
OUT_CSV = BASE / "results" / "indicvoices_genuine.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV.parent.mkdir(exist_ok=True)

# ── Stream and accumulate per-speaker buffers ─────────────────────────────────
print("Loading ai4bharat/IndicVoices (Hindi, streaming)…")
ds = load_dataset(
    "ai4bharat/IndicVoices",
    "hindi",
    split="train",
    streaming=True,
    token=HF_TOKEN,
)

# speaker_id → list of float32 arrays (already at TARGET_SR)
speaker_buffers: dict[str, list[np.ndarray]] = defaultdict(list)
speaker_dur:     dict[str, float]             = defaultdict(float)

saved = []
seen  = 0

for item in ds:
    seen += 1

    spk = str(item.get("speaker_id") or "unknown")
    dur = float(item.get("duration", 0) or 0)

    # audio field in this dataset is "audio_filepath"
    audio_data = item.get("audio_filepath") or item.get("audio")
    if audio_data is None or not isinstance(audio_data, dict):
        continue

    arr = np.array(audio_data["array"], dtype=np.float32)
    sr  = audio_data["sampling_rate"]

    if sr != TARGET_SR:
        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
        sr  = TARGET_SR

    speaker_buffers[spk].append(arr)
    speaker_dur[spk] += len(arr) / sr

    # Once a speaker has enough audio, flush to a WAV
    if speaker_dur[spk] >= TARGET_DUR:
        combined = np.concatenate(speaker_buffers[spk])
        combined = combined[: TARGET_DUR * TARGET_SR]

        idx   = len(saved) + 1
        fname = f"indicvoices_hindi_{idx:03d}.wav"
        sf.write(str(OUT_DIR / fname), combined, TARGET_SR, subtype="PCM_16")

        saved.append({"file": fname, "speaker": spk})
        print(f"  [{idx:2d}/{N_FILES}] {fname}  speaker={spk}")

        # Reset this speaker's buffer
        del speaker_buffers[spk]
        del speaker_dur[spk]

        if len(saved) >= N_FILES:
            break

    if seen % 1000 == 0:
        print(f"  scanned {seen} items, saved {len(saved)}, "
              f"tracking {len(speaker_buffers)} speakers…")

print(f"\nScanned {seen} items, saved {len(saved)} clips → {OUT_DIR}\n")

if len(saved) < 2:
    print("Not enough clips — dataset may not have enough per-speaker audio.")
    sys.exit(1)

# ── Noise analysis ────────────────────────────────────────────────────────────
print(f"Running noise analysis  (IS_thr={IS_THRESHOLD}  SFM_thr={SFM_THRESHOLD})\n")

FIELDS = [
    "file", "speaker", "n_segs", "max_is", "mean_is",
    "sfm_max_dev", "sfm_triggered", "is_manipulated", "verdict",
]

rows = []
genuine = manipulated = inconc = 0

for i, meta in enumerate(saved, 1):
    path = OUT_DIR / meta["file"]
    r    = analyze_file(str(path), segment_duration=2.0,
                        threshold=IS_THRESHOLD, sfm_threshold=SFM_THRESHOLD)

    if r.get("is_manipulated") is True:
        pred = "MANIPULATED";  manipulated += 1
    elif r.get("is_manipulated") is False:
        pred = "GENUINE";      genuine     += 1
    else:
        pred = "INCONCLUSIVE"; inconc      += 1

    print(f"[{i:2d}/{len(saved)}] {meta['file']}  "
          f"IS={r.get('max_divergence', float('nan')):.4f}  "
          f"SFM={r.get('sfm_max_dev', float('nan')):.4f}  → {pred}")

    rows.append({
        "file":           meta["file"],
        "speaker":        meta["speaker"],
        "n_segs":         r.get("n_valid_segments", 0),
        "max_is":         round(r.get("max_divergence",  float("nan")), 6),
        "mean_is":        round(r.get("mean_divergence", float("nan")), 6),
        "sfm_max_dev":    round(r.get("sfm_max_dev",     float("nan")), 6),
        "sfm_triggered":  r.get("sfm_triggered", False),
        "is_manipulated": r.get("is_manipulated"),
        "verdict":        r.get("verdict", ""),
    })

# ── Save CSV ──────────────────────────────────────────────────────────────────
with open(OUT_CSV, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)

# ── Summary ───────────────────────────────────────────────────────────────────
total = len(rows)
print(f"\n{'='*60}")
print(f"Total        : {total}")
print(f"GENUINE      : {genuine}  ({100*genuine/total:.1f}%)")
print(f"MANIPULATED  : {manipulated}  ← false positives")
print(f"INCONCLUSIVE : {inconc}")
print(f"\nCSV → {OUT_CSV}")
print(f"{'='*60}")
