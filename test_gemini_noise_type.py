"""
Test Gemini noise type identification on noise addition files.
Picks files with a single dominant noise type so we can verify accuracy.
Runs full pipeline (CNN sliding window → Gemini) and compares result vs ground truth.

Usage:
  GEMINI_API_KEY="AIza..." python3.11 test_gemini_noise_type.py
"""

import os, io, csv, time
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from google import genai
from google.genai import types

BASE      = Path(os.environ.get("NOISE_BASE_DIR", str(Path(__file__).parent)))
MODEL_PT  = BASE / "models" / "spectrogram_cnn_seg_v2.pt"
ADD_DIR   = BASE / "datasets" / "rasa_manipulated" / "noise_addition"
OUT_CSV   = BASE / "results" / "gemini_noise_type_test.csv"
OUT_CSV.parent.mkdir(exist_ok=True)

SR        = 22050
N_MELS    = 128
FMAX      = 8000
N_FFT     = 2048
HOP       = 512
WIN_LEN   = 30
WIN_STEP  = 10
THRESH    = 0.5

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise SystemExit("Set GEMINI_API_KEY environment variable first.")

# Noise code → full name mapping
CODE_MAP = {"hv": "hvac", "cr": "crowd", "ra": "rain",
            "ou": "outdoor", "hu": "human", "wn": "white_noise"}

# ── Model ─────────────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model  = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(str(MODEL_PT), map_location=device, weights_only=True))
model.to(device); model.eval()

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Gemini client ─────────────────────────────────────────────────────────────
client = genai.Client(api_key=GEMINI_API_KEY)
with open(BASE / "vlm_system_prompt.md") as f:
    SYSTEM_PROMPT = f.read()


# ── Helpers ───────────────────────────────────────────────────────────────────
def make_spec_img(audio_chunk):
    S    = librosa.feature.melspectrogram(y=audio_chunk, sr=SR, n_mels=N_MELS,
                                          fmax=FMAX, n_fft=N_FFT, hop_length=HOP)
    S_db = librosa.power_to_db(S, ref=np.max)
    fig, axes = plt.subplots(2, 1, figsize=(6, 4),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("black")
    axes[0].imshow(S_db, aspect="auto", origin="lower", cmap="magma", vmin=-80, vmax=0)
    axes[0].axis("off")
    floor = S_db[:20, :]
    axes[1].imshow(floor, aspect="auto", origin="lower", cmap="inferno",
                   vmin=np.percentile(floor, 5), vmax=np.percentile(floor, 95))
    axes[1].axis("off")
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="black")
    plt.close(); buf.seek(0)
    return Image.open(buf).convert("RGB")


def noise_floor_mean(chunk):
    S    = librosa.feature.melspectrogram(y=chunk, sr=SR, n_mels=N_MELS,
                                          fmax=FMAX, n_fft=N_FFT, hop_length=HOP)
    return librosa.power_to_db(S, ref=np.max)[:20, :].mean()


def cnn_pred(img):
    x = val_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1)[0]
    return p.argmax().item(), p[1].item()


def merge_windows(flagged):
    if not flagged:
        return []
    sw = sorted(flagged, key=lambda x: x[0])
    m  = [list(sw[0])]
    for s, e, c in sw[1:]:
        if s <= m[-1][1]:
            m[-1][1] = max(m[-1][1], e); m[-1][2] = max(m[-1][2], c)
        else:
            m.append([s, e, c])
    return [tuple(x) for x in m]


def call_gemini(audio, r_start, r_end, delay=7):
    """Send spectrogram to Gemini. delay=7s keeps us under 10 RPM free tier."""
    time.sleep(delay)
    s   = int(r_start * SR); e = int(r_end * SR)
    img = make_spec_img(audio[s:e])
    buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)

    for model_name in ["gemini-2.5-flash", "gemini-2.0-flash"]:
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=[
                    SYSTEM_PROMPT,
                    types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"),
                ],
            )
            return resp.text.strip(), model_name
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                continue
            raise
    return "ERROR: both models unavailable", "none"


def parse_gemini(raw):
    noise_type = confidence = "unknown"
    for line in raw.splitlines():
        if line.startswith("NOISE_TYPE:"):
            noise_type = line.split(":", 1)[1].strip()
        elif line.startswith("CONFIDENCE:"):
            confidence = line.split(":", 1)[1].strip()
    return noise_type, confidence


def true_noise_from_filename(stem):
    """Parse dominant noise type from filename like rasa_..._multi_hv_no_ou_no."""
    parts = stem.split("_multi_")
    if len(parts) < 2:
        return "unknown"
    codes = parts[1].split("_")
    non_no = [CODE_MAP.get(c, c) for c in codes if c != "no"]
    if not non_no:
        return "none"
    # Return most common; if tie, return all
    from collections import Counter
    count = Counter(non_no)
    dominant = count.most_common(1)[0][0]
    return dominant


# ── Pick test files — 2 per noise type ───────────────────────────────────────
# Only files where exactly one segment has noise (easier to evaluate)
target_files = []
needed = {"hvac": 2, "crowd": 2, "rain": 2, "outdoor": 2, "human": 2, "white_noise": 2}

all_wavs = sorted(ADD_DIR.glob("*.wav"))
for wav in all_wavs:
    stem  = wav.stem
    parts = stem.split("_multi_")
    if len(parts) < 2:
        continue
    codes    = parts[1].split("_")
    non_no   = [c for c in codes if c != "no"]
    # Exactly one non-no segment for clean evaluation
    if len(non_no) != 1:
        continue
    noise_type = CODE_MAP.get(non_no[0], non_no[0])
    if needed.get(noise_type, 0) > 0:
        target_files.append((wav, noise_type))
        needed[noise_type] -= 1
    if all(v == 0 for v in needed.values()):
        break

print(f"Selected {len(target_files)} files  ({len(set(n for _,n in target_files))} noise types)")
print(f"Device  : {device}\n")
print(f"{'File':<45} {'True':>12} {'Gemini':>12} {'Conf':>8} {'Match':>6}")
print("─" * 90)

rows = []
correct = 0

for wav, true_type in target_files:
    audio, _ = librosa.load(str(wav), sr=SR, mono=True)
    total_dur = len(audio) / SR
    baseline  = noise_floor_mean(audio)

    # Sliding window CNN
    flagged = []
    for ws in range(0, int(total_dur) - WIN_LEN + 1, WIN_STEP):
        chunk = audio[int(ws*SR):int((ws+WIN_LEN)*SR)]
        lbl, conf = cnn_pred(make_spec_img(chunk))
        if lbl == 1 and conf > THRESH:
            flagged.append((ws, ws + WIN_LEN, conf))

    if not flagged:
        gemini_type = "missed_by_cnn"
        gemini_conf = "-"
        model_used  = "-"
        raw_response = ""
    else:
        merged  = merge_windows(flagged)
        r_start, r_end, _ = merged[0]
        # Only call Gemini if floor rose (addition)
        region  = audio[int(r_start*SR):int(r_end*SR)]
        floor_dir = "rose" if noise_floor_mean(region) - baseline > 2.0 else "dropped"
        if floor_dir == "rose":
            raw_response, model_used = call_gemini(audio, r_start, r_end)
            gemini_type, gemini_conf = parse_gemini(raw_response)
        else:
            gemini_type = "floor_dropped"
            gemini_conf = "-"
            model_used  = "-"
            raw_response = ""

    match = "✓" if gemini_type == true_type else "✗"
    if gemini_type == true_type:
        correct += 1

    print(f"  {wav.name:<43} {true_type:>12} {gemini_type:>12} {gemini_conf:>8} {match:>6}")

    rows.append({
        "filename":    wav.name,
        "true_type":   true_type,
        "gemini_type": gemini_type,
        "confidence":  gemini_conf,
        "match":       match,
    })

total = len(target_files)
print(f"\nAccuracy : {correct}/{total}  ({correct/total*100:.1f}%)")

# Save CSV
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["filename","true_type","gemini_type","confidence","match"])
    w.writeheader(); w.writerows(rows)
print(f"CSV saved: {OUT_CSV}")
