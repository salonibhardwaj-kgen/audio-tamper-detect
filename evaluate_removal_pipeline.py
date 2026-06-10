"""
Evaluate the sliding-window pipeline on:
  - All genuine Rasa files          (~100)
  - All noisereduce removal files   (400)
  - All audacity removal files      (400)

Output CSV: results/removal_pipeline_eval.csv
Columns:
  filename, true_category, true_label, predicted_label, correct,
  windows_flagged, manipulation_type, region_start, region_end,
  exact_boundary_start, exact_boundary_end, position, confidence
"""

import csv
import io
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

# ── Config ──────────────────────────────────────────────────────────────────
BASE     = Path(os.environ.get("NOISE_BASE_DIR", str(Path(__file__).parent)))
MODEL_PT = BASE / "models" / "spectrogram_cnn_seg_v2.pt"
OUT_CSV  = BASE / "results" / "removal_pipeline_eval.csv"
OUT_CSV.parent.mkdir(exist_ok=True)

SR        = 22050
N_MELS    = 128
FMAX      = 8000
N_FFT     = 2048
HOP       = 512
WIN_LEN   = 30
WIN_STEP  = 10
CNN_THRESH = 0.5

# ── Load model ───────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model  = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(str(MODEL_PT), map_location=device,
                                 weights_only=True))
model.to(device)
model.eval()

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Helpers ───────────────────────────────────────────────────────────────────
def make_spectrogram_img(audio_chunk):
    S    = librosa.feature.melspectrogram(y=audio_chunk, sr=SR,
                                          n_mels=N_MELS, fmax=FMAX,
                                          n_fft=N_FFT, hop_length=HOP)
    S_db = librosa.power_to_db(S, ref=np.max)
    fig, axes = plt.subplots(2, 1, figsize=(6, 4),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("black")
    axes[0].imshow(S_db, aspect="auto", origin="lower",
                   cmap="magma", vmin=-80, vmax=0)
    axes[0].axis("off")
    floor = S_db[:20, :]
    axes[1].imshow(floor, aspect="auto", origin="lower", cmap="inferno",
                   vmin=np.percentile(floor, 5),
                   vmax=np.percentile(floor, 95))
    axes[1].axis("off")
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="black")
    plt.close()
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def noise_floor_energy(audio_chunk):
    S    = librosa.feature.melspectrogram(y=audio_chunk, sr=SR,
                                          n_mels=N_MELS, fmax=FMAX,
                                          n_fft=N_FFT, hop_length=HOP)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db[:20, :].mean()


def cnn_predict(pil_img):
    x = val_tf(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out   = model(x)
        probs = torch.softmax(out, dim=1)[0]
    return probs.argmax().item(), probs[1].item()


def merge_windows(flagged):
    if not flagged:
        return []
    sw = sorted(flagged, key=lambda x: x[0])
    merged = [list(sw[0])]
    for s, e, c in sw[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
            merged[-1][2] = max(merged[-1][2], c)
        else:
            merged.append([s, e, c])
    return [tuple(m) for m in merged]


def floor_direction(audio, region_start, region_end, baseline_floor):
    s = int(region_start * SR)
    e = int(region_end   * SR)
    chunk = audio[s:e]
    if len(chunk) < SR:
        return "unknown"
    return "dropped" if noise_floor_energy(chunk) - baseline_floor < -2.0 else "rose"


def change_point(audio, region_start, region_end, frame_sec=1.0):
    frame = int(frame_sec * SR)
    s     = int(region_start * SR)
    e     = int(min(region_end * SR, len(audio)))
    seg   = audio[s:e]
    floors, times = [], []
    for i in range(0, len(seg) - frame, frame // 2):
        floors.append(noise_floor_energy(seg[i:i + frame]))
        times.append(region_start + i / SR)
    if len(floors) < 3:
        return region_start, region_end
    floors = np.array(floors)
    threshold = np.median(floors[:2]) - 5.0
    start_idx = end_idx = None
    for i, f in enumerate(floors):
        if start_idx is None and f < threshold:
            start_idx = i
        if start_idx is not None and f >= threshold - 2.0:
            end_idx = i
            break
    if start_idx is None:
        return region_start, region_end
    es = times[start_idx]
    ee = times[end_idx] if end_idx is not None else times[-1]
    return round(es, 1), round(ee, 1)


def classify_position(r_start, r_end, total_dur):
    mid = (r_start + r_end) / 2
    if mid < total_dur * 0.25:
        return "start"
    elif mid > total_dur * 0.75:
        return "end"
    elif total_dur * 0.35 <= mid <= total_dur * 0.65:
        return "mid"
    else:
        return "random"


# ── Derive ground truth from filename ────────────────────────────────────────
def ground_truth(wav_path: Path):
    stem = wav_path.stem
    if "removal" in stem:
        cat = "noisereduce_" + stem.split("_removal_")[-1]
        return 1, cat
    elif "audacity" in stem:
        cat = "audacity_" + stem.split("_audacity_")[-1]
        return 1, cat
    else:
        return 0, "genuine"


# ── Run pipeline on one file ──────────────────────────────────────────────────
def run_file(wav_path: Path):
    audio, _  = librosa.load(str(wav_path), sr=SR, mono=True)
    total_dur = len(audio) / SR
    baseline  = noise_floor_energy(audio)

    flagged = []
    for win_start in range(0, int(total_dur) - WIN_LEN + 1, WIN_STEP):
        win_end = win_start + WIN_LEN
        chunk   = audio[int(win_start * SR):int(win_end * SR)]
        label, conf = cnn_predict(make_spectrogram_img(chunk))
        if label == 1 and conf > CNN_THRESH:
            flagged.append((win_start, win_end, conf))

    if not flagged:
        return dict(predicted_label=0, windows_flagged=0,
                    manipulation_type="", region_start="", region_end="",
                    exact_boundary_start="", exact_boundary_end="",
                    position="", confidence="")

    merged = merge_windows(flagged)
    r_start, r_end, max_conf = merged[0]   # use largest region if multiple

    direction = floor_direction(audio, r_start, r_end, baseline)

    if direction == "dropped":
        manip = "noise_removal"
        es, ee = change_point(audio, r_start, r_end)
        pos    = classify_position(r_start, r_end, total_dur)
    elif direction == "rose":
        manip = "noise_addition"
        es, ee = r_start, r_end
        pos    = classify_position(r_start, r_end, total_dur)
    else:
        manip = "uncertain"
        es, ee = r_start, r_end
        pos    = classify_position(r_start, r_end, total_dur)

    return dict(predicted_label=1,
                windows_flagged=len(flagged),
                manipulation_type=manip,
                region_start=round(r_start, 1),
                region_end=round(r_end, 1),
                exact_boundary_start=round(es, 1),
                exact_boundary_end=round(ee, 1),
                position=pos,
                confidence=round(max_conf, 4))


# ── Build file list ───────────────────────────────────────────────────────────
RASA   = BASE / "datasets" / "rasa"
NR_DIR = BASE / "datasets" / "rasa_manipulated" / "noise_removal"
AUD_DIR = BASE / "datasets" / "rasa_manipulated" / "audacity_removal"

files = (sorted(RASA.glob("*.wav")) +
         sorted(NR_DIR.glob("*.wav")) +
         sorted(AUD_DIR.glob("*.wav")))

print(f"Total files : {len(files)}")
print(f"Device      : {device}")
print(f"Output CSV  : {OUT_CSV}\n")

# ── Run & write CSV ───────────────────────────────────────────────────────────
FIELDS = ["filename", "true_category", "true_label", "predicted_label",
          "correct", "windows_flagged", "manipulation_type",
          "region_start", "region_end",
          "exact_boundary_start", "exact_boundary_end",
          "position", "confidence"]

correct_total = 0

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()

    for i, wav in enumerate(files, 1):
        true_label, true_cat = ground_truth(wav)
        try:
            result = run_file(wav)
        except Exception as e:
            print(f"  ERROR {wav.name}: {e}")
            result = dict(predicted_label=-1, windows_flagged=0,
                          manipulation_type="ERROR", region_start="",
                          region_end="", exact_boundary_start="",
                          exact_boundary_end="", position="",
                          confidence="")

        pred    = result["predicted_label"]
        correct = int(pred == true_label)
        correct_total += correct

        row = {"filename": wav.name,
               "true_category": true_cat,
               "true_label": true_label,
               **result,
               "correct": correct}
        writer.writerow(row)

        if i % 50 == 0 or i == len(files):
            acc = correct_total / i * 100
            print(f"  {i:>4} / {len(files)}  |  accuracy so far: {acc:.1f}%")

print(f"\nDone.")
print(f"Overall accuracy : {correct_total}/{len(files)} = {correct_total/len(files)*100:.1f}%")
print(f"CSV saved        : {OUT_CSV}")
