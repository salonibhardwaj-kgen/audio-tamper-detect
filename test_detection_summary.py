"""
Synthetic vs Genuine detection summary across languages and manipulation types.
No Gemini API needed — CNN + noise floor direction only.
"""

import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from pathlib import Path
import sys, os

os.environ["GEMINI_API_KEY"] = ""
sys.path.insert(0, str(Path("/Users/salonibhardwaj/Desktop/Noise ")))

BASE     = Path(os.environ.get("NOISE_BASE_DIR", str(Path(__file__).parent)))
RASA     = BASE / "datasets" / "rasa"
NR_WAV   = BASE / "datasets" / "rasa_manipulated" / "noise_removal"
AUD_WAV  = BASE / "datasets" / "rasa_manipulated" / "audacity_removal"
ADD_WAV  = BASE / "datasets" / "rasa_manipulated" / "noise_addition"
NOISE    = BASE / "datasets" / "noise_sources"
SR       = 22050

import importlib.util
spec = importlib.util.spec_from_file_location("pipeline", BASE / "run_new_pipeline.py")
pipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipe)

results = []

def run(label, wav_path, expected):
    verdict = "UNKNOWN"
    manip   = "—"
    try:
        audio, sr = librosa.load(str(wav_path), sr=SR, mono=True)
        total_dur = len(audio) / sr

        baseline_floor = pipe.noise_floor_energy(audio, sr)
        flagged = []
        clean_chunks = []
        for ws in range(0, int(total_dur) - pipe.WIN_LEN + 1, pipe.WIN_STEP):
            s, e = int(ws*sr), int((ws+pipe.WIN_LEN)*sr)
            img = pipe.make_spectrogram_png(audio[s:e], sr)
            lbl, conf = pipe.cnn_predict(img)
            if lbl == 1 and conf > pipe.CNN_THRESH:
                flagged.append((ws, ws+pipe.WIN_LEN, conf))
            else:
                clean_chunks.append(audio[s:e])

        if not flagged:
            verdict = "GENUINE"
            manip   = "—"
        else:
            verdict = "SYNTHETIC"
            merged  = pipe.merge_windows(flagged)
            r_start, r_end, _ = merged[0]
            direction = pipe.noise_floor_direction(audio, sr, r_start, r_end, baseline_floor)
            manip = "REMOVAL" if direction == "dropped" else "ADDITION" if direction == "rose" else "UNCERTAIN"

    except Exception as ex:
        verdict = f"ERROR: {ex!s:.40}"

    correct = "✓" if verdict == expected else "✗"
    results.append((label, expected, verdict, manip, correct))
    print(f"  {correct}  {label:<45} expected={expected:<10} got={verdict:<10} type={manip}")


print("\n── HINDI (training language) ─────────────────────────────────────────")
run("Hindi genuine 001",          RASA / "rasa_Hindi_Male_001.wav",                        "GENUINE")
run("Hindi genuine 002",          RASA / "rasa_Hindi_Male_002.wav",                        "GENUINE")
run("Hindi noisereduce start",    NR_WAV / "rasa_Hindi_Male_001_removal_start.wav",         "SYNTHETIC")
run("Hindi noisereduce mid",      NR_WAV / "rasa_Hindi_Male_001_removal_mid.wav",           "SYNTHETIC")
run("Hindi noisereduce end",      NR_WAV / "rasa_Hindi_Male_001_removal_end.wav",           "SYNTHETIC")
run("Hindi audacity start",       AUD_WAV / "rasa_Hindi_Male_001_audacity_start.wav",       "SYNTHETIC")
run("Hindi audacity mid",         AUD_WAV / "rasa_Hindi_Male_001_audacity_mid.wav",         "SYNTHETIC")
run("Hindi audacity end",         AUD_WAV / "rasa_Hindi_Male_001_audacity_end.wav",         "SYNTHETIC")
run("Hindi noise addition",       ADD_WAV / "rasa_Hindi_Male_001_multi_cr_ou_hv_no.wav",    "SYNTHETIC")

print("\n── ASSAMESE (out-of-distribution) ────────────────────────────────────")
run("Assamese genuine",           RASA / "rasa_Assamese_Male_001.wav",                     "GENUINE")
run("Assamese noisereduce mid",   BASE / "datasets" / "rasa_assamese_noisereduce_mid.wav", "SYNTHETIC")

print("\n── BENGALI (out-of-distribution) ─────────────────────────────────────")
BENGALI = BASE / "datasets" / "rasa_bengali"
run("Bengali combined genuine",   BENGALI / "rasa_Bengali_combined_120s.wav",              "GENUINE")
run("Bengali noisereduce mid",    BENGALI / "rasa_Bengali_combined_noisereduce_mid.wav",   "SYNTHETIC")

# ── On-the-fly noise addition for Assamese ────────────────────────────────
print("\n── ON-THE-FLY NOISE ADDITION (Assamese) ──────────────────────────────")
crowd = librosa.load(str(NOISE / "crowd" / "crowd_murmur.wav"), sr=SR, mono=True)[0]
crowd = crowd / (np.max(np.abs(crowd)) + 1e-9)

audio, _ = librosa.load(str(RASA / "rasa_Assamese_Male_001.wav"), sr=SR, mono=True)
s, e     = int(45*SR), int(75*SR)
n        = e - s
chunk    = crowd[:n]
rms_s    = np.sqrt(np.mean(audio[s:e]**2)) + 1e-9
rms_n    = np.sqrt(np.mean(chunk**2)) + 1e-9
mixed    = audio.copy()
mixed[s:e] = np.clip(audio[s:e] + chunk*(rms_s/rms_n)*0.55, -1, 1)
add_path = BASE / "datasets" / "rasa_assamese_crowd_addition_mid.wav"
sf.write(str(add_path), mixed, SR)
run("Assamese crowd addition mid", add_path, "SYNTHETIC")

# ── Summary ───────────────────────────────────────────────────────────────
correct_n = sum(1 for r in results if r[4] == "✓")
total_n   = len(results)

print(f"\n{'─'*70}")
print(f"  Overall: {correct_n}/{total_n} correct  ({correct_n/total_n*100:.0f}%)")
print(f"{'─'*70}")

by_lang = {"Hindi": [], "Assamese": [], "Bengali": []}
for label, exp, got, manip, correct in results:
    for lang in by_lang:
        if lang.lower() in label.lower():
            by_lang[lang].append(correct == "✓")

for lang, vals in by_lang.items():
    if vals:
        print(f"  {lang:<12}: {sum(vals)}/{len(vals)}")
