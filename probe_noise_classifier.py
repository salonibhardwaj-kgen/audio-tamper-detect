"""
Step 1 + Step 2: Noise type classifier probe.

Step 1: Build spectral templates for each noise type from real source files.
Step 2: Add HVAC noise to first 30s of one genuine clip, classify each 2s
        segment and report per-segment labels + confidence scores.
"""

import sys, csv
import numpy as np
import librosa
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from noise_analyzer import FRAME_LENGTH, HOP_LENGTH

BASE      = Path(os.environ.get("NOISE_BASE_DIR", str(Path(__file__).parent)))
NOISE_DIR = BASE / "datasets" / "noise_sources"
RASA_DIR  = BASE / "datasets" / "rasa"
ESC50_DIR = BASE / "datasets" / "ESC-50-master"

TARGET_SR = 22050
SEG_DUR   = 2.0      # seconds per segment
SNR_DB    = 15.0     # added noise level relative to speech
DEV_THR   = 0.05     # IS deviation from baseline to classify a segment


# ── Noise floor (no SFM guard — needed for white noise templates) ─────────────
def noise_floor_raw(segment):
    if len(segment) < FRAME_LENGTH:
        return None
    frames = librosa.util.frame(segment, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    if frames.shape[1] < 4:
        return None
    power = np.abs(np.fft.rfft(frames, axis=0)) ** 2
    nf = np.percentile(power, 10, axis=1)
    total = float(np.sum(nf))
    if total < 1e-12:
        return None
    return (nf / total).astype(np.float64)


# ── IS divergence (P observed, Q reference) ───────────────────────────────────
def is_div(P, Q):
    P = np.maximum(P, 1e-12)
    Q = np.maximum(Q, 1e-12)
    r = P / Q
    return float(np.mean(r - np.log(r) - 1.0))


# ── Build template: mean noise floor across all segments of audio ─────────────
def build_template(audio):
    seg = int(SEG_DUR * TARGET_SR)
    floors = []
    for i in range(len(audio) // seg):
        nf = noise_floor_raw(audio[i*seg:(i+1)*seg])
        if nf is not None:
            floors.append(nf)
    return np.mean(floors, axis=0) if floors else None


# ── Load and concatenate ESC-50 categories ─────────────────────────────────────
def load_esc50(categories):
    meta  = list(csv.DictReader(open(ESC50_DIR / "meta" / "esc50.csv")))
    clips = [r for r in meta if r["category"] in categories]
    arrs  = []
    for r in clips:
        a, _ = librosa.load(str(ESC50_DIR / "audio" / r["filename"]),
                            sr=TARGET_SR, mono=True)
        arrs.append(a)
    return np.concatenate(arrs) if arrs else np.array([])


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Build templates
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1 — Building noise type templates")
print("=" * 60)

templates = {}

hvac1, _ = librosa.load(str(NOISE_DIR/"hvac"/"hvac_exhaust_fan.wav"),  sr=TARGET_SR, mono=True)
hvac2, _ = librosa.load(str(NOISE_DIR/"hvac"/"hvac_ventilation.wav"),  sr=TARGET_SR, mono=True)
templates["hvac"]    = build_template(np.concatenate([hvac1, hvac2]))

wn, _    = librosa.load(str(NOISE_DIR/"white_noise"/"white_noise.wav"), sr=TARGET_SR, mono=True)
templates["white"]   = build_template(wn[:120*TARGET_SR])

c1, _    = librosa.load(str(NOISE_DIR/"crowd"/"crowd_murmur.wav"),     sr=TARGET_SR, mono=True)
c2, _    = librosa.load(str(NOISE_DIR/"crowd"/"crowd_congenial.wav"),   sr=TARGET_SR, mono=True)
templates["crowd"]   = build_template(np.concatenate([c1, c2]))

templates["human"]   = build_template(load_esc50(["breathing","coughing","snoring"]))
templates["outdoor"] = build_template(load_esc50(["wind","chirping_birds","sea_waves"]))
templates["rain"]    = build_template(load_esc50(["rain","thunderstorm"]))

for name, tmpl in templates.items():
    status = "OK" if tmpl is not None else "FAILED"
    print(f"  {name:<10} {status}")

valid_tmpl = {k: v for k, v in templates.items() if v is not None}
print(f"\n  {len(valid_tmpl)}/6 templates ready\n")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Single clip probe
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 2 — Single clip probe: HVAC added to 0–30s at SNR=15dB")
print("=" * 60)

clip_path = sorted(RASA_DIR.glob("rasa_Hindi_Male_*.wav"))[0]
print(f"\n  Clip : {clip_path.name}")

audio, sr = librosa.load(str(clip_path), sr=TARGET_SR, mono=True)
audio = audio[:120*TARGET_SR]

# Build 30s HVAC noise and scale to SNR
hvac_full = np.concatenate([hvac1, hvac2])
hvac_30s  = np.tile(hvac_full, 5)[:30*TARGET_SR]
window    = audio[:30*TARGET_SR]
rms_s = float(np.sqrt(np.mean(window**2)))
rms_n = float(np.sqrt(np.mean(hvac_30s**2)))
scale = rms_s / max(rms_n, 1e-8) * 10**(-SNR_DB / 20)
mixed = audio.copy()
mixed[:30*TARGET_SR] = window + hvac_30s * scale
print(f"  HVAC scale factor : {scale:.4f}  (SNR={SNR_DB}dB)")

# Compute baseline: median noise floor across all 60 segments of the mixed clip
seg_samples = int(SEG_DUR * TARGET_SR)
n_segs = len(mixed) // seg_samples

all_floors = []
for i in range(n_segs):
    nf = noise_floor_raw(mixed[i*seg_samples:(i+1)*seg_samples])
    if nf is not None:
        all_floors.append(nf)
baseline = np.median(all_floors, axis=0)
print(f"  Baseline from {len(all_floors)}/{n_segs} valid segments\n")

# ── Per-segment classification ─────────────────────────────────────────────────
print(f"  {'Seg':>3}  {'Time':>6}  {'Dev':>7}  {'Label':<10}  "
      + "  ".join(f"{k:<6}" for k in valid_tmpl))
print("  " + "-"*80)

segment_labels = []
segment_devs   = []

for i in range(n_segs):
    seg = mixed[i*seg_samples:(i+1)*seg_samples]
    nf  = noise_floor_raw(seg)
    t   = i * SEG_DUR

    if nf is None:
        segment_labels.append("invalid")
        segment_devs.append(0.0)
        print(f"  {i:>3}  {t:>5.1f}s  {'N/A':>7}  {'invalid':<10}")
        continue

    dev = is_div(nf, baseline)
    segment_devs.append(dev)

    if dev < DEV_THR:
        label = "room"
        segment_labels.append("room")
        prob_str = "  ".join(f"{'—':>6}" for _ in valid_tmpl)
        marker = ""
    else:
        scores = {k: 1.0 / (1.0 + is_div(nf, tmpl)) for k, tmpl in valid_tmpl.items()}
        total  = sum(scores.values())
        probs  = {k: v/total for k, v in scores.items()}
        label  = max(probs, key=probs.get)
        segment_labels.append(label)
        prob_str = "  ".join(f"{probs[k]:>6.2f}" for k in valid_tmpl)
        marker = "  ← expected HVAC" if i < 15 else ""

    print(f"  {i:>3}  {t:>5.1f}s  {dev:>7.4f}  {label:<10}  {prob_str}{marker}")

# ── Summary ────────────────────────────────────────────────────────────────────
valid_labels   = [l for l in segment_labels if l != "invalid"]
label_counts   = Counter(valid_labels)
majority_label = label_counts.most_common(1)[0][0]
manip_score    = 1.0 - label_counts[majority_label] / len(valid_labels)

non_room = [(i, segment_labels[i]) for i in range(len(segment_labels))
            if segment_labels[i] not in ("invalid", majority_label)]

# How many of the expected HVAC segments (0–14) were correctly labelled?
hvac_segs     = [segment_labels[i] for i in range(15)]
hvac_correct  = sum(1 for l in hvac_segs if l == "hvac")
room_segs     = [segment_labels[i] for i in range(15, n_segs)]
room_correct  = sum(1 for l in room_segs if l in ("room", majority_label))

print(f"\n{'='*60}")
print(f"  Majority label      : {majority_label}")
print(f"  Label distribution  : {dict(label_counts)}")
print(f"  Manipulation score  : {manip_score:.2f}  (0=genuine, 1=fully manipulated)")
print(f"  HVAC segs correct   : {hvac_correct}/15  (segs 0–14, expected HVAC)")
print(f"  Room segs correct   : {room_correct}/45  (segs 15–59, expected room)")
print(f"  Non-majority segs   : {non_room[:10]}{'...' if len(non_room)>10 else ''}")
print(f"\n  Verdict : {'MANIPULATED — HVAC detected at 0–30s' if manip_score > 0.1 else 'GENUINE'}")
print(f"{'='*60}")
