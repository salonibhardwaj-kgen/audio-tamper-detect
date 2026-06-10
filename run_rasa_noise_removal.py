"""
Rasa Hindi Male — manipulated audio via selective noise removal.

Takes genuine Rasa clips and applies denoising to a 30s portion:
  - start  : seconds 0–30
  - mid    : seconds 45–75
  - end    : seconds 90–120
  - random : a random 30s window

WAVs  → datasets/rasa_manipulated/noise_removal/
CSV   → results/rasa_noise_removal.csv
"""

import sys, csv, os, random
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import soundfile as sf
import librosa
import noisereduce as nr
from pathlib import Path

from noise_analyzer import analyze_audio, IS_THRESHOLD, SFM_THRESHOLD, RELATIVE_K

# ── Config ─────────────────────────────────────────────────────────────────────
TARGET_SR    = 22050
DENOISE_DUR  = 30        # seconds of audio to denoise
TOTAL_DUR    = 120       # total clip duration

BASE         = Path(os.environ.get("NOISE_BASE_DIR", str(Path(__file__).parent)))
GENUINE_DIR  = BASE / "datasets" / "rasa"
OUT_DIR      = BASE / "datasets" / "rasa_manipulated" / "noise_removal"
OUT_CSV      = BASE / "results" / "rasa_noise_removal.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV.parent.mkdir(exist_ok=True)

FIELDS = [
    "file", "source_clip", "removal_type",
    "denoise_start_s", "denoise_end_s",
    "n_segs", "max_is", "mean_is", "row_ratio",
    "sfm_max_dev", "is_manipulated", "verdict",
]

REMOVAL_TYPES = ["start", "mid", "end", "random"]

# ── Get genuine clips ──────────────────────────────────────────────────────────
all_clips = sorted(GENUINE_DIR.glob("rasa_Hindi_Male_*.wav"))
print(f"Found {len(all_clips)} genuine clips.\n")

# Assign each clip to exactly one removal type
# start: 0–24, mid: 25–49, end: 50–74, random: 75–98
assignments = (
    [(c, "start")  for c in all_clips[0:25]] +
    [(c, "mid")    for c in all_clips[25:50]] +
    [(c, "end")    for c in all_clips[50:75]] +
    [(c, "random") for c in all_clips[75:99]]
)

rows = []

for clip_path, removal_type in assignments:
    audio, sr = librosa.load(str(clip_path), sr=TARGET_SR, mono=True)

    if len(audio) < TOTAL_DUR * TARGET_SR:
        print(f"  SKIP {clip_path.name} — too short")
        continue

    audio = audio[: TOTAL_DUR * TARGET_SR]   # trim to exactly 120s
    denoise_samples = DENOISE_DUR * TARGET_SR

    if True:
        # Determine which 30s window to denoise
        if removal_type == "start":
            d_start = 0
        elif removal_type == "mid":
            d_start = (TOTAL_DUR // 2) - (DENOISE_DUR // 2)   # 45s
        elif removal_type == "end":
            d_start = TOTAL_DUR - DENOISE_DUR                  # 90s
        else:  # random
            max_start = TOTAL_DUR - DENOISE_DUR
            d_start = random.randint(0, max_start)

        d_end     = d_start + DENOISE_DUR
        s_start   = d_start * TARGET_SR
        s_end     = d_end   * TARGET_SR

        # Apply denoising to the selected 30s window only
        manipulated = audio.copy()
        segment     = manipulated[s_start:s_end]
        denoised    = nr.reduce_noise(y=segment, sr=TARGET_SR, stationary=False)
        manipulated[s_start:s_end] = denoised

        # Save WAV
        fname = f"{clip_path.stem}_removal_{removal_type}.wav"
        out_path = OUT_DIR / fname
        sf.write(str(out_path), manipulated, TARGET_SR, subtype="PCM_16")

        # Analyze
        r = analyze_audio(manipulated, TARGET_SR, segment_duration=2.0,
                          threshold=IS_THRESHOLD, sfm_threshold=SFM_THRESHOLD,
                          relative_k=RELATIVE_K)

        pred = ("MANIPULATED" if r.get("is_manipulated") is True
                else "GENUINE"   if r.get("is_manipulated") is False
                else "INCONCLUSIVE")

        print(f"  {fname}  row_ratio={r.get('row_outlier_ratio', float('nan')):.2f}"
              f"  denoise={d_start}–{d_end}s  → {pred}")

        rows.append({
            "file":           fname,
            "source_clip":    clip_path.name,
            "removal_type":   removal_type,
            "denoise_start_s": d_start,
            "denoise_end_s":   d_end,
            "n_segs":         r.get("n_valid_segments", 0),
            "max_is":         round(r.get("max_divergence",    float("nan")), 6),
            "mean_is":        round(r.get("mean_divergence",   float("nan")), 6),
            "row_ratio":      round(r.get("row_outlier_ratio", float("nan")), 4),
            "sfm_max_dev":    round(r.get("sfm_max_dev",       float("nan")), 6),
            "is_manipulated": r.get("is_manipulated"),
            "verdict":        r.get("verdict", ""),
        })

    # Flush CSV after each clip
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV updated ({len(rows)}/99) → {OUT_CSV}\n")

# ── Summary ────────────────────────────────────────────────────────────────────
total       = len(rows)
manipulated = sum(1 for r in rows if r["is_manipulated"] is True  or r["is_manipulated"] == "True")
genuine     = sum(1 for r in rows if r["is_manipulated"] is False or r["is_manipulated"] == "False")
inconc      = total - manipulated - genuine

print(f"\n{'='*60}")
print(f"Total clips   : {total}")
print(f"MANIPULATED   : {manipulated}  ({100*manipulated/total:.1f}%)  ← correctly detected")
print(f"GENUINE (FN)  : {genuine}   ← false negatives (missed)")
print(f"INCONCLUSIVE  : {inconc}")
print(f"\nCSV → {OUT_CSV}")
print(f"{'='*60}")
