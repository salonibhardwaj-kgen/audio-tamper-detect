"""
Large-scale validation of CNN v3.
Tests all available pre-made files — genuine, NR removal, audacity, noise addition.
Saves results to CSV.
"""

import numpy as np
import librosa
import csv
from pathlib import Path
import sys, os, time

os.environ["GEMINI_API_KEY"] = ""
sys.path.insert(0, str(Path(__file__).parent))

BASE    = Path(os.environ.get("NOISE_BASE_DIR", str(Path(__file__).parent)))
RASA    = BASE / "datasets" / "rasa"
NR_WAV  = BASE / "datasets" / "rasa_manipulated" / "noise_removal"
AUD_WAV = BASE / "datasets" / "rasa_manipulated" / "audacity_removal"
ADD_WAV = BASE / "datasets" / "rasa_manipulated" / "noise_addition"
AS_DIR  = BASE / "datasets" / "rasa_assamese"
SR      = 22050

import importlib.util
spec = importlib.util.spec_from_file_location("pipeline", BASE / "run_new_pipeline.py")
pipe = importlib.util.module_from_spec(spec); spec.loader.exec_module(pipe)

def detect(wav_path):
    audio, sr = librosa.load(str(wav_path), sr=SR, mono=True)
    total_dur  = len(audio) / sr
    baseline   = pipe.noise_floor_energy(audio, sr)
    flagged    = []
    for ws in range(0, int(total_dur) - pipe.WIN_LEN + 1, pipe.WIN_STEP):
        s, e = int(ws*sr), int((ws+pipe.WIN_LEN)*sr)
        img  = pipe.make_spectrogram_png(audio[s:e], sr)
        lbl, conf = pipe.cnn_predict(img)
        if lbl == 1 and conf > pipe.CNN_THRESH:
            flagged.append((ws, ws+pipe.WIN_LEN, conf))
    if not flagged:
        return "GENUINE", "—", 0.0
    merged = pipe.merge_windows(flagged)
    r_start, r_end, max_conf = merged[0]
    direction = pipe.noise_floor_direction(audio, sr, r_start, r_end, baseline)
    manip = "REMOVAL" if direction=="dropped" else "ADDITION" if direction=="rose" else "UNCERTAIN"
    return "SYNTHETIC", manip, round(max_conf, 3)

rows    = []
correct = 0
total   = 0

def run_batch(files, expected, category, language):
    global correct, total
    batch_correct = 0
    for i, wav in enumerate(sorted(files)):
        v, m, conf = detect(wav)
        ok = (v == expected)
        if ok: correct += 1; batch_correct += 1
        total += 1
        rows.append({
            "file": wav.name, "category": category, "language": language,
            "expected": expected, "got": v, "manip_type": m,
            "confidence": conf, "correct": "YES" if ok else "NO"
        })
        if (i+1) % 50 == 0 or (i+1) == len(files):
            print(f"    {i+1}/{len(files)}  |  batch accuracy so far: "
                  f"{batch_correct/(i+1)*100:.1f}%")
    acc = batch_correct / len(files) * 100
    print(f"  → {category} ({language}): {batch_correct}/{len(files)} = {acc:.1f}%\n")
    return batch_correct, len(files)

print("Large-scale CNN v3 validation")
print("=" * 55)
t0 = time.time()

print("\n[1/5] Genuine — Hindi (101 files)")
run_batch(list(RASA.glob("rasa_Hindi_Male_*.wav")) +
          list(RASA.glob("rasa_Assamese_Male_*.wav")),
          "GENUINE", "genuine", "Hindi+Assamese")

print("[2/5] NR Removal — Hindi (400 files)")
run_batch(list(NR_WAV.glob("*.wav")),
          "SYNTHETIC", "NR_removal", "Hindi")

print("[3/5] Audacity Removal — Hindi (400 files)")
run_batch(list(AUD_WAV.glob("*.wav")),
          "SYNTHETIC", "audacity_removal", "Hindi")

print("[4/5] Noise Addition — Hindi (399 files)")
run_batch(list(ADD_WAV.glob("*.wav")),
          "SYNTHETIC", "noise_addition", "Hindi")

print("[5/5] Assamese (genuine + on-the-fly NR removal)")
import soundfile as sf, noisereduce as nr
TMP = BASE / "datasets" / "_tmp_ls.wav"
as_files_genuine = list(AS_DIR.glob("rasa_Assamese_Male_0*.wav"))
run_batch(as_files_genuine, "GENUINE", "genuine", "Assamese")

# On-the-fly NR removal for Assamese
as_manip = []
for wav in sorted(as_files_genuine):
    audio, _ = librosa.load(str(wav), sr=SR, mono=True)
    seg = audio[45*SR:75*SR]
    red = nr.reduce_noise(y=seg, sr=SR, y_noise=seg[:SR], prop_decrease=1.0, stationary=False)
    out = audio.copy(); out[45*SR:75*SR] = red
    sf.write(str(TMP), out, SR)
    v, m, conf = detect(TMP)
    ok = (v == "SYNTHETIC")
    if ok: correct += 1
    total += 1
    rows.append({"file": wav.name+"_nr_mid", "category": "NR_removal",
                 "language": "Assamese", "expected": "SYNTHETIC", "got": v,
                 "manip_type": m, "confidence": conf, "correct": "YES" if ok else "NO"})
    print(f"    {'✓' if ok else '✗'}  {wav.name}  →  {v}  ({m})")
TMP.unlink(missing_ok=True)

# ── Summary ───────────────────────────────────────────────────────────────────
elapsed = time.time() - t0
print(f"\n{'='*55}")
print(f"  OVERALL : {correct}/{total} = {correct/total*100:.2f}%")
print(f"  Time    : {elapsed/60:.1f} min")
print(f"{'='*55}\n")

by_cat = {}
for r in rows:
    k = r["category"]
    by_cat.setdefault(k, []).append(r["correct"] == "YES")
for cat, vals in by_cat.items():
    print(f"  {cat:<25}: {sum(vals)}/{len(vals)} ({sum(vals)/len(vals)*100:.1f}%)")

# ── Save CSV ──────────────────────────────────────────────────────────────────
out_csv = BASE / "results" / "cnn_v3_large_scale_results.csv"
out_csv.parent.mkdir(exist_ok=True)
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print(f"\n  CSV saved: {out_csv}  ({len(rows)} rows)")
