"""
AI4Bharat Rasa — genuine recording baseline test.

Rasa has 2 professional speakers (Male + Female) per language.
Strategy: stream Hindi Male in one pass, saving a new 120s WAV every time
the buffer fills. Hindi Male has 20+ hours so we can get 100 non-overlapping
clips from a single speaker in a single studio — perfect stationarity.

Probe confirmed: Hindi Male → GENUINE (row_ratio=3.59, SFM_dev=0.1986).

Requires: export HF_TOKEN=hf_...
WAVs  → datasets/rasa/
CSV   → results/rasa_genuine.csv

State file (datasets/rasa/stream_state.json) tracks how many Male utterances
were consumed in the last streaming pass so resuming skips them and collects
only new audio. If the state file is missing and WAVs exist, the script aborts
rather than silently producing duplicates.
"""

import sys, csv, os, json
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from datasets import load_dataset

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("Set HF_TOKEN first:  export HF_TOKEN=hf_...")

from noise_analyzer import analyze_audio, IS_THRESHOLD, SFM_THRESHOLD, RELATIVE_K

# ── Config ─────────────────────────────────────────────────────────────────────
TARGET_SR  = 22050
TARGET_DUR = 120
N_FILES    = 100
BATCH_SIZE = 5
LANGUAGE   = "Hindi"
GENDER     = "Male"

BASE       = Path(os.environ.get("NOISE_BASE_DIR", str(Path(__file__).parent)))
OUT_DIR    = BASE / "datasets" / "rasa"
OUT_CSV    = BASE / "results" / "rasa_genuine.csv"
STATE_FILE = OUT_DIR / "stream_state.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV.parent.mkdir(exist_ok=True)

FIELDS = [
    "file", "language", "gender",
    "n_segs", "max_is", "mean_is", "row_ratio",
    "sfm_max_dev", "sfm_triggered", "is_manipulated", "verdict",
]

# ── State file helpers ──────────────────────────────────────────────────────────
def load_utterances_consumed() -> int:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text()).get("utterances_consumed", 0)
    return 0

def save_utterances_consumed(n: int) -> None:
    STATE_FILE.write_text(json.dumps({"utterances_consumed": n}))

# ── Resume — filter strictly to this language + gender ─────────────────────────
existing = sorted(OUT_DIR.glob(f"rasa_{LANGUAGE}_{GENDER}_*.wav"))
saved: list[dict] = [
    {"file": p.name, "language": LANGUAGE, "gender": GENDER}
    for p in existing
]
existing_rows: list[dict] = []
if OUT_CSV.exists():
    with open(OUT_CSV, newline="") as fh:
        existing_rows = list(csv.DictReader(fh))

# Guard: without a state file we cannot know the stream position and would
# silently collect duplicate audio. Abort with clear instructions.
prev_consumed = load_utterances_consumed()
if not STATE_FILE.exists() and saved:
    print(f"ERROR: {len(saved)} {LANGUAGE}/{GENDER} clips exist in {OUT_DIR}")
    print(f"       but no state file found at {STATE_FILE}.")
    print()
    print("Options:")
    print(f"  A) Delete all WAVs and start fresh:")
    print(f"       rm {OUT_DIR}/rasa_{LANGUAGE}_{GENDER}_*.wav")
    print(f"  B) Create the state file manually (set N = Male utterances")
    print(f"     consumed in the last run, or a safe over-estimate):")
    print(f"       echo '{{\"utterances_consumed\": N}}' > {STATE_FILE}")
    raise SystemExit("Aborted — cannot resume safely without state file.")

if saved:
    print(f"Resuming: {len(saved)} clips already saved.")
    print(f"Stream state: {prev_consumed} Male utterances consumed last run.")
    print()

all_rows     = list(existing_rows)
new_in_batch: list[dict] = []

# ── Next available filename index ──────────────────────────────────────────────
def next_clip_idx() -> int:
    """Return lowest integer not already used as a clip filename."""
    used = set()
    for p in OUT_DIR.glob(f"rasa_{LANGUAGE}_{GENDER}_*.wav"):
        try:
            used.add(int(p.stem.split("_")[-1]))
        except ValueError:
            pass
    idx = 1
    while idx in used:
        idx += 1
    return idx

# ── Batch analysis + CSV flush ─────────────────────────────────────────────────
def flush_batch(batch: list[dict]) -> None:
    if not batch:
        return
    print(f"\n{'─'*55}")
    print(f"Analysing batch of {len(batch)} clips…")
    g = m = ic = 0
    new_rows = []
    for meta in batch:
        path = OUT_DIR / meta["file"]
        audio, sr = librosa.load(str(path), sr=None, mono=True)
        r = analyze_audio(audio, sr, segment_duration=2.0,
                          threshold=IS_THRESHOLD, sfm_threshold=SFM_THRESHOLD,
                          relative_k=RELATIVE_K)

        pred = ("MANIPULATED" if r.get("is_manipulated") is True
                else "GENUINE"   if r.get("is_manipulated") is False
                else "INCONCLUSIVE")
        if pred == "GENUINE":       g  += 1
        elif pred == "MANIPULATED": m  += 1
        else:                       ic += 1

        idx = len(all_rows) + len(new_rows) + 1
        print(f"  [{idx:2d}] {meta['file']}  "
              f"IS={r.get('max_divergence', float('nan')):.4f}  "
              f"row={r.get('row_outlier_ratio', float('nan')):.2f}  "
              f"SFM={r.get('sfm_max_dev', float('nan')):.4f}  → {pred}")

        new_rows.append({
            "file":           meta["file"],
            "language":       meta["language"],
            "gender":         meta["gender"],
            "n_segs":         r.get("n_valid_segments", 0),
            "max_is":         round(r.get("max_divergence",  float("nan")), 6),
            "mean_is":        round(r.get("mean_divergence", float("nan")), 6),
            "row_ratio":      round(r.get("row_outlier_ratio", float("nan")), 4),
            "sfm_max_dev":    round(r.get("sfm_max_dev",     float("nan")), 6),
            "sfm_triggered":  r.get("sfm_triggered", False),
            "is_manipulated": r.get("is_manipulated"),
            "verdict":        r.get("verdict", ""),
        })

    all_rows.extend(new_rows)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rows)

    total_so_far   = len(all_rows)
    genuine_so_far = sum(1 for r in all_rows
                         if r["is_manipulated"] is False
                         or r["is_manipulated"] == "False")
    print(f"\nBatch  — GENUINE: {g}  MANIPULATED: {m}  INCONCLUSIVE: {ic}")
    print(f"Total  — {total_so_far} clips  |  genuine so far: {genuine_so_far}/{total_so_far}")
    print(f"CSV updated → {OUT_CSV}")
    print(f"{'─'*55}\n")
    batch.clear()

# ── Single streaming pass: collect N_FILES non-overlapping 120s clips ──────────
need = N_FILES - len(saved)
if need <= 0:
    print("Already have enough clips — running analysis only.\n")
else:
    print(f"Streaming {LANGUAGE} / {GENDER}  —  collecting {need} more clips…")
    if prev_consumed > 0:
        print(f"Skipping first {prev_consumed} Male utterances (already used)…\n")
    else:
        print()

    try:
        ds = load_dataset("ai4bharat/Rasa", LANGUAGE,
                          split="train", streaming=True, token=HF_TOKEN)
    except Exception as e:
        raise SystemExit(f"Failed to load dataset: {e}")

    buffer:       list[np.ndarray] = []
    buf_dur       = 0.0
    male_utt_seen = 0   # Male utterances seen in this streaming pass (from shard 0)

    for item in ds:
        if len(saved) >= N_FILES:
            break

        if str(item.get("gender", "")) != GENDER:
            continue

        male_utt_seen += 1

        # Skip utterances already consumed in the previous run
        if male_utt_seen <= prev_consumed:
            if male_utt_seen % 500 == 0:
                print(f"  Skipping: {male_utt_seen}/{prev_consumed}…")
            continue

        audio_field = item.get("audio")
        if not isinstance(audio_field, dict) or "array" not in audio_field:
            continue

        arr = np.array(audio_field["array"], dtype=np.float32)
        sr  = audio_field["sampling_rate"]
        if sr != TARGET_SR:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)

        clip_dur   = len(arr) / TARGET_SR
        buffer.append(arr)
        buf_dur   += clip_dur

        style = str(item.get("style", ""))
        text  = str(item.get("text", ""))[:55]
        print(f"  utt {male_utt_seen:5d}  +{clip_dur:.2f}s  buf={buf_dur:.1f}s  "
              f"[{style}]  '{text}'")

        if buf_dur >= TARGET_DUR:
            combined = np.concatenate(buffer)[: TARGET_DUR * TARGET_SR]
            idx      = next_clip_idx()
            fname    = f"rasa_{LANGUAGE}_{GENDER}_{idx:03d}.wav"
            sf.write(str(OUT_DIR / fname), combined, TARGET_SR, subtype="PCM_16")
            meta = {"file": fname, "language": LANGUAGE, "gender": GENDER}
            saved.append(meta)
            new_in_batch.append(meta)
            print(f"\n  ✓ Saved clip {len(saved)}/{N_FILES}: {fname}\n")

            buffer  = []
            buf_dur = 0.0

            if len(new_in_batch) >= BATCH_SIZE:
                flush_batch(new_in_batch)

    # Persist stream position: male_utt_seen is total Male utterances from shard 0
    # encountered in this pass (both skipped and newly processed).
    save_utterances_consumed(male_utt_seen)
    print(f"Stream state saved: {male_utt_seen} Male utterances consumed total.")

# Flush last partial batch
if new_in_batch:
    flush_batch(new_in_batch)

# ── Final summary ──────────────────────────────────────────────────────────────
total       = len(all_rows)
genuine     = sum(1 for r in all_rows if r["is_manipulated"] is False
                  or r["is_manipulated"] == "False")
manipulated = sum(1 for r in all_rows if r["is_manipulated"] is True
                  or r["is_manipulated"] == "True")
inconc      = total - genuine - manipulated

print(f"\n{'='*60}")
print(f"Dataset      : Rasa {LANGUAGE} {GENDER}")
print(f"Total        : {total}")
if total:
    print(f"GENUINE      : {genuine}  ({100*genuine/total:.1f}%)")
print(f"MANIPULATED  : {manipulated}  ← false positives")
print(f"INCONCLUSIVE : {inconc}")
print(f"\nCSV → {OUT_CSV}")
print(f"{'='*60}")
