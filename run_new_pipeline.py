"""
New sliding-window inference pipeline.

Flow:
  1. Load audio (any duration)
  2. Generate full-clip spectrogram as baseline
  3. Slide 30s window every 10s → spectrogram per window → CNN
  4. Merge flagged windows → suspicious time ranges
  5. Noise floor direction (vs full-clip baseline) → removal or addition
  6. If removal: change point detection → exact time + start/mid/end/random
  7. If addition: Gemini 2.5 Flash VLM → noise type + time range

Model: models/spectrogram_cnn_seg_v2.pt  (ResNet18, genuine vs synthetic)

Set GEMINI_API_KEY below (get free key at aistudio.google.com).
Leave as empty string "" to skip VLM and report location only.
"""

import os
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
from pathlib import Path

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ── Config ──────────────────────────────────────────────────────────────────
BASE      = Path(os.environ.get("NOISE_BASE_DIR", str(Path(__file__).parent)))
MODEL_PT  = BASE / "models" / "spectrogram_cnn_seg_v3.pt"

SR        = 22050
N_MELS    = 128
FMAX      = 8000
N_FFT     = 2048
HOP       = 512

WIN_LEN   = 30    # seconds per window
WIN_STEP  = 10    # step between windows
CNN_THRESH = 0.5  # confidence threshold for synthetic

# ── Load model ──────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(str(MODEL_PT), map_location=device, weights_only=True))
model.to(device)
model.eval()

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Spectrogram helpers ─────────────────────────────────────────────────────
def make_spectrogram_png(audio_segment, sr=SR):
    """Return 2-panel mel spectrogram as an in-memory PNG (PIL Image)."""
    S    = librosa.feature.melspectrogram(y=audio_segment, sr=sr,
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


def make_diff_spectrogram_png(manip_segment, baseline_segment, sr=SR):
    """
    Difference spectrogram: manipulated − baseline using a shared reference.
    Positive values = energy added by noise. Shows pure noise fingerprint.
    """
    def compute_mel(audio):
        S = librosa.feature.melspectrogram(y=audio, sr=sr,
                                           n_mels=N_MELS, fmax=FMAX,
                                           n_fft=N_FFT, hop_length=HOP)
        return S

    S_manip = compute_mel(manip_segment)
    S_base  = compute_mel(baseline_segment)

    # Resize baseline columns to match manipulated if lengths differ
    if S_base.shape[1] != S_manip.shape[1]:
        from PIL import Image as PILImage
        base_img = PILImage.fromarray(S_base.astype(np.float32))
        S_base   = np.array(base_img.resize((S_manip.shape[1], S_manip.shape[0]),
                                             PILImage.BILINEAR))

    # Use shared max for consistent dB scale across both
    global_max   = max(np.max(S_manip), np.max(S_base))
    manip_db     = librosa.power_to_db(S_manip, ref=global_max)
    base_db      = librosa.power_to_db(S_base,  ref=global_max)

    # Difference: only keep positive (added noise), clip negatives to 0
    diff_db      = np.maximum(manip_db - base_db, 0)

    fig, axes = plt.subplots(2, 1, figsize=(6, 4),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("black")
    vmax = max(np.percentile(diff_db, 99), 1.0)
    axes[0].imshow(diff_db, aspect="auto", origin="lower",
                   cmap="magma", vmin=0, vmax=vmax)
    axes[0].axis("off")
    floor = diff_db[:20, :]
    floor_vmax = max(np.percentile(floor, 99), 0.5)
    axes[1].imshow(floor, aspect="auto", origin="lower", cmap="inferno",
                   vmin=0, vmax=floor_vmax)
    axes[1].axis("off")
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="black")
    plt.close()
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def noise_floor_energy(audio_segment, sr=SR):
    """Mean energy of bottom 20 mel bins — proxy for noise floor level."""
    S    = librosa.feature.melspectrogram(y=audio_segment, sr=sr,
                                          n_mels=N_MELS, fmax=FMAX,
                                          n_fft=N_FFT, hop_length=HOP)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db[:20, :].mean()


# ── CNN prediction ───────────────────────────────────────────────────────────
def cnn_predict(pil_img):
    """Return (label, confidence).  label 0=genuine, 1=synthetic."""
    x = val_tf(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out   = model(x)
        probs = torch.softmax(out, dim=1)[0]
    label = probs.argmax().item()
    conf  = probs[1].item()   # confidence of synthetic
    return label, conf


# ── Merge overlapping windows ────────────────────────────────────────────────
def merge_windows(flagged_windows):
    """
    flagged_windows: list of (start_s, end_s, conf)
    Returns list of merged (start_s, end_s, max_conf)
    """
    if not flagged_windows:
        return []
    sorted_w = sorted(flagged_windows, key=lambda x: x[0])
    merged   = [list(sorted_w[0])]
    for s, e, c in sorted_w[1:]:
        if s <= merged[-1][1]:          # overlaps
            merged[-1][1] = max(merged[-1][1], e)
            merged[-1][2] = max(merged[-1][2], c)
        else:
            merged.append([s, e, c])
    return [tuple(m) for m in merged]


# ── Noise floor direction ────────────────────────────────────────────────────
def noise_floor_direction(audio, sr, region_start, region_end, baseline_floor):
    """
    Compare noise floor in suspicious region vs full-clip baseline.
    Returns 'dropped' (removal) or 'rose' (addition).
    """
    s = int(region_start * sr)
    e = int(region_end   * sr)
    region_audio = audio[s:e]
    if len(region_audio) < sr:    # too short to measure reliably
        return "unknown"
    region_floor = noise_floor_energy(region_audio, sr)
    delta = region_floor - baseline_floor
    return "dropped" if delta < -2.0 else "rose"


# ── Change point detection ───────────────────────────────────────────────────
def change_point_detection(audio, sr, region_start, region_end, frame_sec=1.0):
    """
    Scan frame-by-frame noise floor inside suspicious region.
    Returns (exact_start, exact_end) of the manipulation in seconds,
    or (region_start, region_end) if no clear change point found.
    """
    frame = int(frame_sec * sr)
    s     = int(region_start * sr)
    e     = int(min(region_end * sr, len(audio)))
    segment = audio[s:e]

    floors = []
    times  = []
    for i in range(0, len(segment) - frame, frame // 2):
        chunk = segment[i:i + frame]
        floors.append(noise_floor_energy(chunk, sr))
        times.append(region_start + i / sr)

    if len(floors) < 3:
        return region_start, region_end

    floors = np.array(floors)
    baseline_f = np.median(floors[:2])    # first 2 frames as local baseline

    # Find first frame that drops significantly (removal signature)
    threshold = baseline_f - 5.0
    start_idx = None
    end_idx   = None

    for i, f in enumerate(floors):
        if start_idx is None and f < threshold:
            start_idx = i
        if start_idx is not None and f >= threshold - 2.0:
            end_idx = i
            break

    if start_idx is None:
        return region_start, region_end

    exact_start = times[start_idx]
    exact_end   = times[end_idx] if end_idx is not None else times[-1]
    return round(exact_start, 1), round(exact_end, 1)


# ── Classify position ─────────────────────────────────────────────────────────
def classify_position(start_s, end_s, total_dur):
    """Classify where manipulation falls: start / mid / end / random."""
    mid_point = (start_s + end_s) / 2
    if mid_point < total_dur * 0.25:
        return "start"
    elif mid_point > total_dur * 0.75:
        return "end"
    elif total_dur * 0.35 <= mid_point <= total_dur * 0.65:
        return "mid"
    else:
        return "random"


# ── Gemini VLM — noise type identification ──────────────────────────────────
def gemini_identify(audio, sr, r_start, r_end, baseline_audio=None):
    """
    Send difference spectrogram (manipulated − baseline) to Gemini 2.5 Flash.
    baseline_audio: a clean 30s window from the same clip (not flagged by CNN).
                    If None, falls back to the raw spectrogram.
    Returns (noise_type, confidence, precise_start_s, precise_end_s).
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Crop suspicious region
    s   = int(r_start * sr)
    e   = int(r_end   * sr)
    manip_seg = audio[s:e]

    # Use raw spectrogram — Gemini understands real spectrograms, not diff images
    img = make_spectrogram_png(manip_seg, sr)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    with open(BASE / "vlm_system_prompt.md") as f:
        system_prompt = f.read()

    for model_name in ["gemini-2.5-flash", "gemini-2.0-flash"]:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    system_prompt,
                    types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"),
                ],
            )
            break
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                print(f"     [{model_name} unavailable, trying next...]")
                continue
            raise
    else:
        return "unavailable", "unknown", r_start, r_end

    raw = response.text.strip()

    # Parse structured output
    noise_type  = "unknown"
    confidence  = "unknown"
    start_pct   = 0.0
    end_pct     = 100.0

    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("NOISE_TYPE:"):
            noise_type = line.split(":", 1)[1].strip()
        elif line.startswith("CONFIDENCE:"):
            confidence = line.split(":", 1)[1].strip()
        elif line.startswith("TIME_START:"):
            try:
                start_pct = float(line.split(":", 1)[1].strip().rstrip("%"))
            except ValueError:
                pass
        elif line.startswith("TIME_END:"):
            try:
                end_pct = float(line.split(":", 1)[1].strip().rstrip("%"))
            except ValueError:
                pass

    # Convert percentages → absolute timestamps
    region_dur    = r_end - r_start
    precise_start = round(r_start + region_dur * start_pct / 100, 1)
    precise_end   = round(r_start + region_dur * end_pct   / 100, 1)

    return noise_type, confidence, precise_start, precise_end


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
def run_pipeline(wav_path: str):
    wav_path = Path(wav_path)
    print(f"\n{'═'*65}")
    print(f"  FILE : {wav_path.name}")
    print(f"{'═'*65}")

    # 1. Load audio
    audio, sr = librosa.load(str(wav_path), sr=SR, mono=True)
    total_dur = len(audio) / sr
    print(f"  Duration : {total_dur:.1f}s")

    # 2. Full-clip baseline noise floor
    baseline_floor = noise_floor_energy(audio, sr)
    print(f"  Baseline noise floor : {baseline_floor:.2f} dB")

    # 3. Sliding window → CNN per window
    flagged      = []
    clean_chunks = []   # genuine windows used as baseline for diff spectrogram
    n_windows    = 0

    for win_start in range(0, int(total_dur) - WIN_LEN + 1, WIN_STEP):
        win_end = win_start + WIN_LEN
        s = int(win_start * sr)
        e = int(win_end   * sr)
        chunk = audio[s:e]

        img = make_spectrogram_png(chunk, sr)
        label, conf = cnn_predict(img)
        n_windows += 1

        if label == 1 and conf > CNN_THRESH:
            flagged.append((win_start, win_end, conf))
        else:
            clean_chunks.append(chunk)

    # Best baseline: average of all clean windows (reduces speech variation)
    if clean_chunks:
        min_len  = min(len(c) for c in clean_chunks)
        baseline_audio = np.mean([c[:min_len] for c in clean_chunks], axis=0)
    else:
        baseline_audio = None

    print(f"  Windows scanned : {n_windows}")
    print(f"  Windows flagged : {len(flagged)}")

    # 4. All genuine?
    if not flagged:
        print("\n  VERDICT : GENUINE  (no manipulation detected)")
        print(f"{'─'*65}")
        return

    # 5. Merge flagged windows → suspicious time ranges
    merged = merge_windows(flagged)
    print(f"  Merged regions  : {len(merged)}")

    # 6. Per-region analysis
    for i, (r_start, r_end, max_conf) in enumerate(merged):
        print(f"\n  ── Region {i+1}: {r_start:.0f}s – {r_end:.0f}s  (conf={max_conf:.3f}) ──")

        direction = noise_floor_direction(audio, sr, r_start, r_end, baseline_floor)
        print(f"     Noise floor direction : {direction.upper()}")

        if direction == "dropped":
            # ── NOISE REMOVAL ──────────────────────────────────────────────
            print("     Manipulation type    : NOISE REMOVAL")
            exact_start, exact_end = change_point_detection(
                audio, sr, r_start, r_end)
            position = classify_position(r_start, r_end, total_dur)
            print(f"     Exact boundary       : {exact_start}s – {exact_end}s")
            print(f"     Position in clip     : {position.upper()}")
            print("     Noise type           : CANNOT IDENTIFY  "
                  "(signal destroyed — original noise no longer exists)")

        elif direction == "rose":
            # ── NOISE ADDITION ─────────────────────────────────────────────
            print("     Manipulation type    : NOISE ADDITION")
            position = classify_position(r_start, r_end, total_dur)
            print(f"     Approximate range    : {r_start:.0f}s – {r_end:.0f}s")
            print(f"     Position in clip     : {position.upper()}")

            if GEMINI_API_KEY:
                try:
                    noise_type, confidence, precise_start, precise_end = gemini_identify(
                        audio, sr, r_start, r_end, baseline_audio)
                    print(f"     Noise type           : {noise_type}")
                    print(f"     Confidence           : {confidence}")
                    print(f"     Precise time range   : {precise_start}s – {precise_end}s")
                except Exception as e:
                    print(f"     Noise type           : [Gemini error: {e!s:.80}]")
            else:
                print("     Noise type           : [set GEMINI_API_KEY to enable]")

        else:
            print("     Manipulation type    : UNCERTAIN  (floor change too small)")

    print(f"\n  VERDICT : SYNTHETIC  (manipulation detected)")
    print(f"{'─'*65}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CASES
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    RASA       = BASE / "datasets" / "rasa"
    NR_WAV     = BASE / "datasets" / "rasa_manipulated" / "noise_removal"
    AUD_WAV    = BASE / "datasets" / "rasa_manipulated" / "audacity_removal"
    ADD_WAV    = BASE / "datasets" / "rasa_manipulated" / "noise_addition"

    test_files = [
        # Genuine
        RASA / "rasa_Hindi_Male_001.wav",
        RASA / "rasa_Hindi_Male_002.wav",

        # Noisereduce removal
        NR_WAV / "rasa_Hindi_Male_001_removal_start.wav",
        NR_WAV / "rasa_Hindi_Male_001_removal_mid.wav",
        NR_WAV / "rasa_Hindi_Male_001_removal_end.wav",

        # Audacity removal
        AUD_WAV / "rasa_Hindi_Male_001_audacity_start.wav",
        AUD_WAV / "rasa_Hindi_Male_001_audacity_mid.wav",
        AUD_WAV / "rasa_Hindi_Male_001_audacity_end.wav",

        # Noise addition
        ADD_WAV / "rasa_Hindi_Male_001_multi_cr_ou_hv_no.wav",
    ]

    print(f"Model  : {MODEL_PT.name}")
    print(f"Device : {device}")

    for f in test_files:
        if not f.exists():
            print(f"\n[SKIP] {f.name} — file not found")
            continue
        run_pipeline(str(f))
