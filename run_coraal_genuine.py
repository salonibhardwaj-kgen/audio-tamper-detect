"""
CORAAL (Corpus of Regional African American Language) — genuine baseline test.

Downloads ATL/DCA interview MP3s directly from the University of Oregon server.
Each interview is one speaker in one room (~40 min). We slice non-overlapping
120-second segments from within each interview, so every segment has a perfectly
stationary noise floor (same room, same mic, same time).

Collects 50 × 120s segments across the first few interviews.

WAVs  → datasets/coraal/
CSV   → results/coraal_genuine.csv
"""

import sys, csv, os
sys.path.insert(0, str(Path(__file__).parent))

import urllib.request
import ssl
import tempfile

_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

from noise_analyzer import analyze_audio, IS_THRESHOLD, SFM_THRESHOLD

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL   = "https://lingtools.uoregon.edu/coraal/explorer/files/MP3s/"
TARGET_DUR = 120      # seconds per output clip
TARGET_SR  = 22050
N_FILES    = 50       # clips to collect
SKIP_START = 60       # skip first 60s of each interview (intro/silence)

BASE    = Path(os.environ.get("NOISE_BASE_DIR", str(Path(__file__).parent)))
OUT_DIR = BASE / "datasets" / "coraal"
OUT_CSV = BASE / "results" / "coraal_genuine.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV.parent.mkdir(exist_ok=True)

# ATL interviews: 13 files, each ~40 min, one speaker per file
INTERVIEWS = [
    "ATL_se0_ag1_f_01_1.mp3",
    "ATL_se0_ag1_f_02_1.mp3",
    "ATL_se0_ag1_f_03_1.mp3",
    "ATL_se0_ag1_m_01_1.mp3",
    "ATL_se0_ag1_m_02_1.mp3",
    "ATL_se0_ag2_f_01_1.mp3",
    "ATL_se0_ag2_m_01_1.mp3",
]

# ── Resume: pick up already-saved clips ──────────────────────────────────────
existing = sorted(OUT_DIR.glob("coraal_*.wav"))
saved = [{"file": p.name,
          "speaker": p.stem.rsplit("_", 1)[0],
          "seg": 0, "start_s": 0}
         for p in existing]
if saved:
    print(f"Resuming: {len(saved)} clips already saved, collecting {N_FILES - len(saved)} more.\n")

for interview in INTERVIEWS:
    if len(saved) >= N_FILES:
        break

    url = BASE_URL + interview
    spk = interview.replace(".mp3", "")
    print(f"\nDownloading {interview}…")

    try:
        with urllib.request.urlopen(url, context=_ssl_ctx) as resp:
            mp3_bytes = resp.read()
    except Exception as e:
        print(f"  SKIP: {e}")
        continue

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(mp3_bytes)
        tmp_path = tmp.name

    try:
        audio, sr = librosa.load(tmp_path, sr=TARGET_SR, mono=True)
    except Exception as e:
        print(f"  Load error: {e}")
        os.unlink(tmp_path)
        continue
    finally:
        os.unlink(tmp_path)

    total_dur = len(audio) / sr
    print(f"  Loaded {total_dur:.0f}s  →  slicing into {TARGET_DUR}s segments")

    seg_samples  = TARGET_DUR * TARGET_SR
    start_sample = SKIP_START * TARGET_SR
    seg_idx      = 0

    while start_sample + seg_samples <= len(audio) and len(saved) < N_FILES:
        chunk = audio[start_sample : start_sample + seg_samples]

        idx   = len(saved) + 1
        fname = f"coraal_{idx:03d}.wav"
        sf.write(str(OUT_DIR / fname), chunk, TARGET_SR, subtype="PCM_16")

        saved.append({"file": fname, "speaker": spk, "seg": seg_idx,
                      "start_s": start_sample / TARGET_SR})
        print(f"  [{idx:2d}/{N_FILES}] {fname}  {start_sample/TARGET_SR:.0f}–"
              f"{(start_sample+seg_samples)/TARGET_SR:.0f}s  spk={spk}")

        start_sample += seg_samples
        seg_idx      += 1

print(f"\nSaved {len(saved)} clips → {OUT_DIR}\n")

if len(saved) < 2:
    print("Not enough clips.")
    sys.exit(1)

# ── Noise analysis ────────────────────────────────────────────────────────────
print(f"Running noise analysis  (IS_thr={IS_THRESHOLD}  SFM_thr={SFM_THRESHOLD})\n")

FIELDS = [
    "file", "speaker", "seg", "start_s",
    "n_segs", "max_is", "mean_is", "sfm_max_dev",
    "sfm_triggered", "is_manipulated", "verdict",
]

rows = []
genuine = manipulated = inconc = 0

for i, meta in enumerate(saved, 1):
    path = OUT_DIR / meta["file"]
    audio, sr = librosa.load(str(path), sr=None, mono=True)
    r = analyze_audio(audio, sr, segment_duration=2.0,
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
        "seg":            meta["seg"],
        "start_s":        meta["start_s"],
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
