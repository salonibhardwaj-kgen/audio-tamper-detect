"""
1. Tests pipeline on FLEURS phone/home recordings (truly OOD)
2. Exports full CSV results for CNN v3 accuracy
"""

import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
import csv
from pathlib import Path
import sys, os

os.environ["GEMINI_API_KEY"] = ""
sys.path.insert(0, str(Path(__file__).parent))

BASE    = Path(os.environ.get("NOISE_BASE_DIR", str(Path(__file__).parent)))
RASA    = BASE / "datasets" / "rasa"
NR_WAV  = BASE / "datasets" / "rasa_manipulated" / "noise_removal"
AUD_WAV = BASE / "datasets" / "rasa_manipulated" / "audacity_removal"
ADD_WAV = BASE / "datasets" / "rasa_manipulated" / "noise_addition"
AS_DIR  = BASE / "datasets" / "rasa_assamese"
BN_DIR  = BASE / "datasets" / "rasa_bengali"
FLEURS  = BASE / "datasets" / "fleurs_test"
NOISE   = BASE / "datasets" / "noise_sources"
SR      = 22050

import importlib.util
spec = importlib.util.spec_from_file_location("pipeline", BASE / "run_new_pipeline.py")
pipe = importlib.util.module_from_spec(spec); spec.loader.exec_module(pipe)

crowd = librosa.load(str(NOISE/"crowd"/"crowd_murmur.wav"), sr=SR, mono=True)[0]
crowd = crowd / (np.max(np.abs(crowd)) + 1e-9)

def add_crowd(audio, s_sec=45, e_sec=75, ratio=0.60):
    s, e = int(s_sec*SR), int(e_sec*SR)
    n = e - s
    chunk = np.tile(crowd, (n//len(crowd))+2)[:n]
    rms_s = np.sqrt(np.mean(audio[s:e]**2))+1e-9
    rms_n = np.sqrt(np.mean(chunk**2))+1e-9
    out = audio.copy()
    out[s:e] = np.clip(audio[s:e]+chunk*(rms_s/rms_n)*ratio,-1,1)
    return out

def apply_nr(audio, s_sec=45, e_sec=75):
    s, e = int(s_sec*SR), int(e_sec*SR)
    seg = audio[s:e]
    red = nr.reduce_noise(y=seg, sr=SR, y_noise=seg[:SR], prop_decrease=1.0, stationary=False)
    out = audio.copy(); out[s:e] = red
    return out

TMP = BASE / "datasets" / "_tmp.wav"

def detect(wav_path):
    audio, sr = librosa.load(str(wav_path), sr=SR, mono=True)
    total_dur  = len(audio) / sr
    baseline   = pipe.noise_floor_energy(audio, sr)
    flagged = []
    for ws in range(0, int(total_dur) - pipe.WIN_LEN + 1, pipe.WIN_STEP):
        s, e = int(ws*sr), int((ws+pipe.WIN_LEN)*sr)
        img  = pipe.make_spectrogram_png(audio[s:e], sr)
        lbl, conf = pipe.cnn_predict(img)
        if lbl == 1 and conf > pipe.CNN_THRESH:
            flagged.append((ws, ws+pipe.WIN_LEN, conf))
    if not flagged:
        return "GENUINE", "—", 0
    merged = pipe.merge_windows(flagged)
    r_start, r_end, max_conf = merged[0]
    direction = pipe.noise_floor_direction(audio, sr, r_start, r_end, baseline)
    manip = "REMOVAL" if direction=="dropped" else "ADDITION" if direction=="rose" else "UNCERTAIN"
    return "SYNTHETIC", manip, round(max_conf, 3)

rows = []

def run(label, wav_path, expected, dataset, language, manipulation):
    v, m, conf = detect(wav_path)
    correct = v == expected
    rows.append({
        "label": label, "dataset": dataset, "language": language,
        "manipulation": manipulation, "expected": expected,
        "got": v, "type": m, "confidence": conf,
        "correct": "YES" if correct else "NO"
    })
    mark = "✓" if correct else "✗"
    print(f"  {mark}  {label:<48} {expected:<10} → {v:<10} {m:<10} conf={conf}")
    return correct

print("\n══ CNN v3 DETECTION RESULTS ══════════════════════════════════════════════\n")

print("── HINDI (Rasa studio, training language) ────────────────────────────────")
run("Hindi genuine 001",        RASA/"rasa_Hindi_Male_001.wav",               "GENUINE",   "Rasa","Hindi","none")
run("Hindi genuine 002",        RASA/"rasa_Hindi_Male_002.wav",               "GENUINE",   "Rasa","Hindi","none")
run("Hindi genuine 005",        RASA/"rasa_Hindi_Male_005.wav",               "GENUINE",   "Rasa","Hindi","none")
run("Hindi NR removal start",   NR_WAV/"rasa_Hindi_Male_001_removal_start.wav","SYNTHETIC","Rasa","Hindi","NR_removal")
run("Hindi NR removal mid",     NR_WAV/"rasa_Hindi_Male_001_removal_mid.wav", "SYNTHETIC", "Rasa","Hindi","NR_removal")
run("Hindi NR removal end",     NR_WAV/"rasa_Hindi_Male_001_removal_end.wav", "SYNTHETIC", "Rasa","Hindi","NR_removal")
run("Hindi audacity start",     AUD_WAV/"rasa_Hindi_Male_001_audacity_start.wav","SYNTHETIC","Rasa","Hindi","audacity_removal")
run("Hindi audacity mid",       AUD_WAV/"rasa_Hindi_Male_001_audacity_mid.wav","SYNTHETIC","Rasa","Hindi","audacity_removal")
run("Hindi audacity end",       AUD_WAV/"rasa_Hindi_Male_001_audacity_end.wav","SYNTHETIC","Rasa","Hindi","audacity_removal")
run("Hindi noise addition",     ADD_WAV/"rasa_Hindi_Male_001_multi_cr_ou_hv_no.wav","SYNTHETIC","Rasa","Hindi","noise_addition")

print("\n── ASSAMESE (Rasa studio, out-of-distribution) ───────────────────────────")
run("Assamese genuine 001",     RASA/"rasa_Assamese_Male_001.wav",            "GENUINE",   "Rasa","Assamese","none")
run("Assamese genuine 002",     AS_DIR/"rasa_Assamese_Male_002.wav",          "GENUINE",   "Rasa","Assamese","none")
run("Assamese genuine 003",     AS_DIR/"rasa_Assamese_Male_003.wav",          "GENUINE",   "Rasa","Assamese","none")

as_audio, _ = librosa.load(str(RASA/"rasa_Assamese_Male_001.wav"), sr=SR, mono=True)
sf.write(str(TMP), apply_nr(as_audio), SR)
run("Assamese NR removal mid",  TMP, "SYNTHETIC", "Rasa","Assamese","NR_removal")
sf.write(str(TMP), add_crowd(as_audio), SR)
run("Assamese crowd add mid",   TMP, "SYNTHETIC", "Rasa","Assamese","noise_addition")

print("\n── BENGALI (Rasa studio, out-of-distribution) ────────────────────────────")
run("Bengali combined genuine", BN_DIR/"rasa_Bengali_combined_120s.wav",       "GENUINE",  "Rasa","Bengali","none")
bn_audio, _ = librosa.load(str(BN_DIR/"rasa_Bengali_combined_120s.wav"), sr=SR, mono=True)
sf.write(str(TMP), apply_nr(bn_audio), SR)
run("Bengali NR removal mid",   TMP, "SYNTHETIC", "Rasa","Bengali","NR_removal")
sf.write(str(TMP), add_crowd(bn_audio), SR)
run("Bengali crowd add mid",    TMP, "SYNTHETIC", "Rasa","Bengali","noise_addition")

print("\n── FLEURS HINDI (crowdsourced phone/home, truly OOD) ─────────────────────")
fl_hi, _ = librosa.load(str(FLEURS/"fleurs_hi_in_120s.wav"), sr=SR, mono=True)
run("FLEURS Hindi genuine",     FLEURS/"fleurs_hi_in_120s.wav",               "GENUINE",   "FLEURS","Hindi","none")
sf.write(str(TMP), apply_nr(fl_hi), SR)
run("FLEURS Hindi NR removal",  TMP, "SYNTHETIC", "FLEURS","Hindi","NR_removal")
sf.write(str(TMP), add_crowd(fl_hi), SR)
run("FLEURS Hindi crowd add",   TMP, "SYNTHETIC", "FLEURS","Hindi","noise_addition")

print("\n── FLEURS TAMIL (crowdsourced phone/home, truly OOD) ─────────────────────")
fl_ta, _ = librosa.load(str(FLEURS/"fleurs_ta_in_120s.wav"), sr=SR, mono=True)
run("FLEURS Tamil genuine",     FLEURS/"fleurs_ta_in_120s.wav",               "GENUINE",   "FLEURS","Tamil","none")
sf.write(str(TMP), apply_nr(fl_ta), SR)
run("FLEURS Tamil NR removal",  TMP, "SYNTHETIC", "FLEURS","Tamil","NR_removal")
sf.write(str(TMP), add_crowd(fl_ta), SR)
run("FLEURS Tamil crowd add",   TMP, "SYNTHETIC", "FLEURS","Tamil","noise_addition")

print("\n── FLEURS BENGALI (crowdsourced phone/home, truly OOD) ───────────────────")
fl_bn, _ = librosa.load(str(FLEURS/"fleurs_bn_in_120s.wav"), sr=SR, mono=True)
run("FLEURS Bengali genuine",   FLEURS/"fleurs_bn_in_120s.wav",               "GENUINE",   "FLEURS","Bengali","none")
sf.write(str(TMP), apply_nr(fl_bn), SR)
run("FLEURS Bengali NR removal",TMP, "SYNTHETIC", "FLEURS","Bengali","NR_removal")
sf.write(str(TMP), add_crowd(fl_bn), SR)
run("FLEURS Bengali crowd add", TMP, "SYNTHETIC", "FLEURS","Bengali","noise_addition")

TMP.unlink(missing_ok=True)

# ── Summary ───────────────────────────────────────────────────────────────────
total   = len(rows)
correct = sum(1 for r in rows if r["correct"]=="YES")

print(f"\n{'═'*70}")
print(f"  Overall: {correct}/{total} ({correct/total*100:.1f}%)")
print(f"{'═'*70}")

# by dataset
for ds in ["Rasa", "FLEURS"]:
    sub = [r for r in rows if r["dataset"]==ds]
    c   = sum(1 for r in sub if r["correct"]=="YES")
    print(f"  {ds:<10}: {c}/{len(sub)}")

# by manipulation type
print()
for manip in ["none", "NR_removal", "audacity_removal", "noise_addition"]:
    sub = [r for r in rows if r["manipulation"]==manip]
    if sub:
        c = sum(1 for r in sub if r["correct"]=="YES")
        print(f"  {manip:<20}: {c}/{len(sub)}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = BASE / "results" / "cnn_v3_full_results.csv"
csv_path.parent.mkdir(exist_ok=True)
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["label","dataset","language","manipulation",
                                       "expected","got","type","confidence","correct"])
    w.writeheader(); w.writerows(rows)
print(f"\n  CSV saved: {csv_path}")
