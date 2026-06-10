"""
Quick probe: AI4Bharat Rasa Hindi — one speaker, one 120s clip.
Tests whether studio TTS recordings pass the noise stationarity analyzer
or get flagged as UNANALYZABLE (broadcast/studio processing).

Requires: export HF_TOKEN=hf_...
Accept terms at: https://huggingface.co/datasets/ai4bharat/Rasa
"""

import sys, os
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset, get_dataset_config_names

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("Set HF_TOKEN first:  export HF_TOKEN=hf_...")

from noise_analyzer import analyze_audio, IS_THRESHOLD, SFM_THRESHOLD, RELATIVE_K

TARGET_SR  = 22050
TARGET_DUR = 120
OUT_DIR    = Path("/Users/salonibhardwaj/Desktop/Noise /datasets/rasa_probe")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Discover available configs ─────────────────────────────────────────────────
print("Fetching Rasa configs…")
try:
    configs = get_dataset_config_names("ai4bharat/Rasa", token=HF_TOKEN)
    print(f"  Configs: {configs[:10]}")
    # Pick Hindi config
    hindi_cfg = next((c for c in configs if c.lower() in ("hi", "hindi", "hin")), None)
    if hindi_cfg is None:
        hindi_cfg = configs[0]
    print(f"  Using config: {hindi_cfg}")
except Exception as e:
    print(f"  Could not list configs ({e}) — trying 'hi'")
    hindi_cfg = "hi"

# ── Peek at one item to understand structure ───────────────────────────────────
print("\nPeeking at dataset structure…")
try:
    ds_peek = load_dataset("ai4bharat/Rasa", hindi_cfg,
                           split="train", streaming=True, token=HF_TOKEN)
    item0 = next(iter(ds_peek))
    print("  Columns:", list(item0.keys()))
    for k, v in item0.items():
        if k == "audio":
            if isinstance(v, dict):
                arr = np.array(v["array"])
                print(f"  audio: sr={v['sampling_rate']}  dur={len(arr)/v['sampling_rate']:.2f}s")
            else:
                print(f"  audio: {type(v)}")
        else:
            print(f"  {k}: {repr(v)[:100]}")
except Exception as e:
    print(f"  Peek failed: {e}")
    sys.exit(1)

# ── Identify speaker key ───────────────────────────────────────────────────────
# Rasa has exactly 2 speakers per language (Male / Female), identified by
# the 'gender' field. 'filename' is the utterance ID, not the speaker.
spk_key = "gender"
target_spk = str(item0.get("gender", "Male"))
print(f"\n  Speaker key: {spk_key}  →  targeting speaker = '{target_spk}'")

# ── Accumulate one speaker's audio until 120s ─────────────────────────────────
print(f"\nAccumulating {TARGET_DUR}s from {target_spk} Hindi speaker…")
buffer: list[np.ndarray] = []
total_dur   = 0.0
clips_used  = 0

ds = load_dataset("ai4bharat/Rasa", hindi_cfg,
                  split="train", streaming=True, token=HF_TOKEN)

for item in ds:
    if str(item.get(spk_key)) != target_spk:
        continue

    audio_field = item.get("audio")
    if not isinstance(audio_field, dict) or "array" not in audio_field:
        continue

    arr = np.array(audio_field["array"], dtype=np.float32)
    sr  = audio_field["sampling_rate"]
    if sr != TARGET_SR:
        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)

    buffer.append(arr)
    total_dur += len(arr) / TARGET_SR
    clips_used += 1

    if total_dur >= TARGET_DUR:
        break

print(f"  Accumulated {total_dur:.1f}s from {clips_used} clips")

if total_dur < 30:
    print("  Not enough audio from this speaker — try a different config.")
    sys.exit(1)

combined = np.concatenate(buffer)[: TARGET_DUR * TARGET_SR]
out_path = OUT_DIR / "rasa_hindi_probe.wav"
sf.write(str(out_path), combined, TARGET_SR, subtype="PCM_16")
print(f"  Saved → {out_path}")

# ── Run analyzer ───────────────────────────────────────────────────────────────
print("\nRunning noise stationarity analysis…")
audio, sr = librosa.load(str(out_path), sr=None, mono=True)
r = analyze_audio(audio, sr, segment_duration=2.0,
                  threshold=IS_THRESHOLD, sfm_threshold=SFM_THRESHOLD,
                  relative_k=RELATIVE_K)

print(f"\n{'='*55}")
print(f"Speaker      : {target_spk}")
print(f"Config       : {hindi_cfg}")
print(f"Duration     : {len(audio)/sr:.1f}s  SR: {sr} Hz")
print(f"Valid segs   : {r.get('n_valid_segments', 'N/A')}")
print(f"Max IS       : {r.get('max_divergence', 'N/A')}")
print(f"Mean IS      : {r.get('mean_divergence', 'N/A')}")
print(f"Row ratio    : {r.get('row_outlier_ratio', 'N/A')}")
print(f"SFM_dev      : {r.get('sfm_max_dev', 'N/A')}")
print(f"Verdict      : {r['verdict']}")
print(f"{'='*55}")

if r.get("is_manipulated") is None:
    print("\nRESULT: UNANALYZABLE — studio/broadcast processing detected.")
    print("  Rasa is too clean for the noise floor estimator.")
    print("  The noise floor SFM is too flat (whitened by studio processing).")
elif not r.get("is_manipulated"):
    print("\nRESULT: GENUINE ✓ — Rasa works as a baseline dataset.")
    print("  Proceed with run_rasa_genuine.py for 50 clips.")
else:
    print("\nRESULT: FALSE POSITIVE — concatenation creating noise inconsistency.")
    print(f"  row_ratio={r.get('row_outlier_ratio'):.2f}, SFM_dev={r.get('sfm_max_dev'):.4f}")
