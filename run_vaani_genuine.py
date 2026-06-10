"""
ARTPARK-IISc Project Vaani — genuine recording baseline test.

Vaani is organized by State_District configs. Each clip is a volunteer
responding to image prompts in their natural environment (home/outdoor),
recorded via mobile app — no telephone codec, no broadcast processing.

Strategy: load one Hindi-speaking district config, group clips by
speaker_id, concatenate until 120s, save as WAV. Since Vaani recordings
are single-session (one volunteer, one sitting), all clips from the same
speaker share the same acoustic environment → stationary noise floor.

Requires: export HF_TOKEN=hf_...
Accept terms at: https://huggingface.co/datasets/ARTPARK-IISc/Vaani

WAVs  → datasets/vaani_hindi/
CSV   → results/vaani_genuine.csv
"""

import sys, csv, os
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset, get_dataset_config_names

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit(
        "Set HF_TOKEN before running:\n"
        "  export HF_TOKEN=hf_...\n"
        "Accept terms at: https://huggingface.co/datasets/ARTPARK-IISc/Vaani"
    )

from noise_analyzer import analyze_audio, IS_THRESHOLD, SFM_THRESHOLD

# ── Config ─────────────────────────────────────────────────────────────────────
TARGET_DUR = 120      # seconds per output clip
TARGET_SR  = 22050
N_FILES    = 50

# Hindi-speaking states to try (in priority order)
HINDI_STATES = [
    "UttarPradesh",
    "Bihar",
    "MadhyaPradesh",
    "Rajasthan",
    "Jharkhand",
    "Uttarakhand",
    "HimachalPradesh",
    "Haryana",
    "Chhattisgarh",
]

BASE    = Path(os.environ.get("NOISE_BASE_DIR", str(Path(__file__).parent)))
OUT_DIR = BASE / "datasets" / "vaani_hindi"
OUT_CSV = BASE / "results" / "vaani_genuine.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV.parent.mkdir(exist_ok=True)

# ── Discover Hindi district configs ───────────────────────────────────────────
print("Fetching available Vaani configs…")
try:
    all_configs = get_dataset_config_names("ARTPARK-IISc/Vaani", token=HF_TOKEN)
    print(f"  {len(all_configs)} total configs found")
except Exception as e:
    print(f"  Could not list configs: {e}")
    all_configs = []

# Filter to Hindi-state districts (config names start with state name)
hindi_configs = []
for state in HINDI_STATES:
    matches = [c for c in all_configs if c.startswith(state)]
    hindi_configs.extend(sorted(matches))

if not hindi_configs:
    # Fallback: try known config names directly
    hindi_configs = [
        "UttarPradesh_Lucknow",
        "UttarPradesh_Varanasi",
        "Bihar_Patna",
        "MadhyaPradesh_Bhopal",
    ]
    print(f"  No configs found via API — trying fallbacks: {hindi_configs[:4]}")
else:
    print(f"  {len(hindi_configs)} Hindi-state configs found: {hindi_configs[:5]}…")

# ── Two-pass streaming per config ─────────────────────────────────────────────
# Pass 1 (fast): stream metadata only — no audio decoding — to find which
#   sessions have >= TARGET_DUR total audio via the 'duration' field.
# Pass 2 (audio): stream again, decode audio only for qualifying sessions.
# This avoids downloading the full dataset (~18 GB for one district config).

saved: list[dict] = []

def stream_metadata(config: str, token: str) -> dict[str, float]:
    """Stream metadata only (no audio access) to tally duration per session."""
    ds = load_dataset("ARTPARK-IISc/Vaani", config,
                      split="train", streaming=True, token=token)
    totals: dict[str, float] = defaultdict(float)
    for item in ds:
        sess = str(item.get("speakerImageHash") or item.get("speakerID") or "unknown")
        dur  = float(item.get("duration") or 0)
        totals[sess] += dur
    return totals

def process_config(config: str) -> int:
    """Two-pass approach: metadata scan then targeted audio download."""
    print(f"\nConfig: {config}")

    # Pass 1 — metadata only (fast, no audio bytes downloaded)
    print("  Pass 1: scanning session durations…", end=" ", flush=True)
    try:
        totals = stream_metadata(config, HF_TOKEN)
    except Exception as e:
        print(f"SKIP ({e})")
        return 0

    qualifying = {s for s, d in totals.items() if d >= TARGET_DUR}
    print(f"{len(totals)} sessions, {len(qualifying)} have >= {TARGET_DUR}s")

    if not qualifying:
        return 0

    # Pass 2 — audio for qualifying sessions only
    print(f"  Pass 2: downloading audio for {len(qualifying)} sessions…")
    try:
        ds2 = load_dataset("ARTPARK-IISc/Vaani", config,
                           split="train", streaming=True, token=HF_TOKEN)
    except Exception as e:
        print(f"  SKIP ({e})")
        return 0

    session_buffers: dict[str, list[np.ndarray]] = defaultdict(list)
    session_spk:     dict[str, str]              = {}
    flushed = 0

    for item in ds2:
        if len(saved) >= N_FILES:
            break
        sess = str(item.get("speakerImageHash") or item.get("speakerID") or "unknown")
        if sess not in qualifying:
            continue  # skip non-qualifying sessions (no audio decode triggered)

        audio_field = item.get("audio")
        if not isinstance(audio_field, dict) or "array" not in audio_field:
            continue

        arr = np.array(audio_field["array"], dtype=np.float32)
        sr  = audio_field["sampling_rate"]
        if sr != TARGET_SR:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)

        session_buffers[sess].append(arr)
        if sess not in session_spk:
            session_spk[sess] = str(item.get("speakerID") or "")

        total_s = sum(len(a) for a in session_buffers[sess]) / TARGET_SR
        if total_s >= TARGET_DUR:
            combined = np.concatenate(session_buffers[sess])[: TARGET_DUR * TARGET_SR]
            idx   = len(saved) + 1
            fname = f"vaani_hindi_{idx:03d}.wav"
            sf.write(str(OUT_DIR / fname), combined, TARGET_SR, subtype="PCM_16")
            saved.append({
                "file":    fname,
                "session": sess,
                "speaker": session_spk[sess],
                "config":  config,
                "state":   config.split("_")[0],
            })
            print(f"    [{idx:2d}/{N_FILES}] {fname}  session={sess[:24]}")
            del session_buffers[sess]
            qualifying.discard(sess)
            flushed += 1

    return flushed

for config in hindi_configs:
    if len(saved) >= N_FILES:
        break
    process_config(config)

print(f"\nSaved {len(saved)} clips → {OUT_DIR}\n")

if len(saved) < 2:
    print("Not enough clips — most Vaani sessions are shorter than TARGET_DUR.")
    print(f"Consider reducing TARGET_DUR (currently {TARGET_DUR}s).")
    sys.exit(1)

# ── Noise analysis ─────────────────────────────────────────────────────────────
print(f"Running noise analysis  (IS_thr={IS_THRESHOLD}  SFM_thr={SFM_THRESHOLD})\n")

FIELDS = [
    "file", "session", "speaker", "config", "state",
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
        "session":        meta["session"],
        "speaker":        meta["speaker"],
        "config":         meta["config"],
        "state":          meta["state"],
        "n_segs":         r.get("n_valid_segments", 0),
        "max_is":         round(r.get("max_divergence",  float("nan")), 6),
        "mean_is":        round(r.get("mean_divergence", float("nan")), 6),
        "sfm_max_dev":    round(r.get("sfm_max_dev",     float("nan")), 6),
        "sfm_triggered":  r.get("sfm_triggered", False),
        "is_manipulated": r.get("is_manipulated"),
        "verdict":        r.get("verdict", ""),
    })

with open(OUT_CSV, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)

total = len(rows)
print(f"\n{'='*60}")
print(f"Total        : {total}")
print(f"GENUINE      : {genuine}  ({100*genuine/total:.1f}%)")
print(f"MANIPULATED  : {manipulated}  ← false positives")
print(f"INCONCLUSIVE : {inconc}")
print(f"\nCSV → {OUT_CSV}")
print(f"{'='*60}")
