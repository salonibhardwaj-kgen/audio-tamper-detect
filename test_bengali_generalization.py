"""
Generalization test on Bengali Rasa audio (out-of-distribution).
Files: rasa_Bengali_001.wav (31.5s), rasa_Bengali_002.wav (36.3s)

Steps:
  1. Run pipeline on both genuine Bengali files   → expect GENUINE
  2. Apply noisereduce to mid portion of each     → create manipulated versions
  3. Run pipeline on manipulated versions         → expect SYNTHETIC + REMOVAL
"""

import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from pathlib import Path
import sys, os

os.environ.setdefault("GEMINI_API_KEY", "")
sys.path.insert(0, str(Path("/Users/salonibhardwaj/Desktop/Noise ")))

BASE        = Path(os.environ.get("NOISE_BASE_DIR", str(Path(__file__).parent)))
BENGALI_DIR = BASE / "datasets" / "rasa_bengali"
SR          = 22050

import importlib.util
spec = importlib.util.spec_from_file_location("pipeline", BASE / "run_new_pipeline.py")
pipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipe)

# Only original files (not previously created noisereduce files)
files = sorted(p for p in BENGALI_DIR.glob("*.wav") if "noisereduce" not in p.name)
print(f"Found {len(files)} original Bengali files")

# Concatenate all clips and tile to ~120s
clips = []
for p in files:
    a, _ = librosa.load(str(p), sr=SR, mono=True)
    clips.append(a)

combined = np.concatenate(clips)
# Tile until we have at least 120s
target   = 120 * SR
if len(combined) < target:
    repeats  = int(np.ceil(target / len(combined)))
    combined = np.tile(combined, repeats)
combined = combined[:target]

genuine_path = BENGALI_DIR / "rasa_Bengali_combined_120s.wav"
sf.write(str(genuine_path), combined, SR)
print(f"Combined genuine clip: {genuine_path.name}  ({len(combined)/SR:.1f}s)\n")

# ── Test 1: Genuine ───────────────────────────────────────────────
print("=" * 65)
print("  TEST 1 — GENUINE Bengali (expect GENUINE verdict)")
print("=" * 65)
pipe.run_pipeline(str(genuine_path))

# ── Apply noisereduce to mid segment only (45–75s) ────────────────
SEG_START = 45
SEG_END   = 75
s = int(SEG_START * SR)
e = int(SEG_END   * SR)
segment       = combined[s:e]
noise_profile = segment[:SR]
reduced       = nr.reduce_noise(y=segment, sr=SR,
                                y_noise=noise_profile,
                                prop_decrease=1.0,
                                stationary=False)

manipulated       = combined.copy()
manipulated[s:e]  = reduced

manip_path = BENGALI_DIR / "rasa_Bengali_combined_noisereduce_mid.wav"
sf.write(str(manip_path), manipulated, SR)

print(f"\n{'='*65}")
print(f"  TEST 2 — NOISEREDUCE REMOVAL at mid ({SEG_START}s–{SEG_END}s)")
print(f"  (expect SYNTHETIC + NOISE REMOVAL + MID)")
print("=" * 65)
pipe.run_pipeline(str(manip_path))

print("\nBengali generalization test complete.")
