"""
Test Gemini VLM on partial noise addition:
  - Add HVAC   to 10s–20s within the 0–30s  segment  (START position)
  - Add CROWD  to 70s–79s within the 60–90s segment  (MID   position)

For each synthetic window:
  1. Generate 30s mel spectrogram
  2. Send to Gemini
  3. Print: detected noise type, confidence, detected time range vs ground truth

Uses free tier — 7s delay between Gemini calls.
"""

import os, io, time
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    GEMINI_API_KEY = input("Enter GEMINI_API_KEY: ").strip()

BASE       = Path(os.environ.get("NOISE_BASE_DIR", str(Path(__file__).parent)))
GENUINE    = BASE / "datasets" / "rasa"
NOISE_DIR  = BASE / "datasets" / "noise_sources"

SR     = 22050
N_MELS = 128
FMAX   = 8000
N_FFT  = 2048
HOP    = 512

# ── Load noise sources ────────────────────────────────────────────────────────
def load_noise(path):
    a, _ = librosa.load(str(path), sr=SR, mono=True)
    return a / (np.max(np.abs(a)) + 1e-9)

crowd_audio      = load_noise(NOISE_DIR / "crowd"       / "crowd_murmur.wav")
rain_audio       = load_noise(NOISE_DIR / "rain"        / "rain_combined.wav")
outdoor_audio    = load_noise(NOISE_DIR / "outdoor"     / "birds_outdoor.wav")
hvac_audio       = load_noise(NOISE_DIR / "hvac"        / "hvac_ventilation.wav")
whitenoise_audio = load_noise(NOISE_DIR / "white_noise" / "white_noise.wav")
human_audio      = load_noise(NOISE_DIR / "human"       / "office_scene.wav")
print("Noise sources loaded.")

def get_chunk(noise, n):
    if len(noise) >= n:
        s = np.random.randint(0, len(noise) - n)
        return noise[s:s+n]
    return np.tile(noise, (n // len(noise)) + 2)[:n]


# ── Mix noise into a sub-window ───────────────────────────────────────────────
def mix_partial(audio, seg_start_s, noise_start_s, noise_end_s, noise_src, mix=0.55):
    """
    audio        : full clip
    seg_start_s  : start of the 30s segment in the full clip
    noise_start_s: absolute time where noise begins
    noise_end_s  : absolute time where noise ends
    """
    out = audio.copy()
    ns  = int(noise_start_s * SR)
    ne  = int(noise_end_s   * SR)
    n   = ne - ns
    chunk = get_chunk(noise_src, n)
    speech_rms = np.sqrt(np.mean(out[ns:ne] ** 2)) + 1e-9
    noise_rms  = np.sqrt(np.mean(chunk ** 2)) + 1e-9
    out[ns:ne] = np.clip(out[ns:ne] + chunk * (speech_rms / noise_rms) * mix, -1, 1)
    return out


# ── Spectrogram helpers ───────────────────────────────────────────────────────
def make_raw_spec(segment):
    """Standard 2-panel mel spectrogram — what Gemini was trained to understand."""
    S    = librosa.feature.melspectrogram(y=segment, sr=SR,
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
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="black")
    plt.close(); buf.seek(0)
    return buf.getvalue()


def make_diff_spec(manip_segment, baseline_segment):
    """Difference spectrogram: isolates only the added noise."""
    def compute_mel(audio):
        return librosa.feature.melspectrogram(y=audio, sr=SR,
                                              n_mels=N_MELS, fmax=FMAX,
                                              n_fft=N_FFT, hop_length=HOP)
    S_manip = compute_mel(manip_segment)
    S_base  = compute_mel(baseline_segment)

    if S_base.shape[1] != S_manip.shape[1]:
        from PIL import Image as PILImage
        base_img = PILImage.fromarray(S_base.astype(np.float32))
        S_base   = np.array(base_img.resize((S_manip.shape[1], S_manip.shape[0]),
                                             PILImage.BILINEAR))

    global_max = max(np.max(S_manip), np.max(S_base))
    manip_db   = librosa.power_to_db(S_manip, ref=global_max)
    base_db    = librosa.power_to_db(S_base,  ref=global_max)
    diff_db    = np.maximum(manip_db - base_db, 0)

    fig, axes = plt.subplots(2, 1, figsize=(6, 4),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("black")
    vmax = max(np.percentile(diff_db, 99), 1.0)
    axes[0].imshow(diff_db, aspect="auto", origin="lower",
                   cmap="magma", vmin=0, vmax=vmax)
    axes[0].axis("off")
    floor = diff_db[:20, :]
    axes[1].imshow(floor, aspect="auto", origin="lower", cmap="inferno",
                   vmin=0, vmax=max(np.percentile(floor, 99), 0.5))
    axes[1].axis("off")
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="black")
    plt.close(); buf.seek(0)
    return buf.getvalue()


# ── Gemini call ───────────────────────────────────────────────────────────────
client = genai.Client(api_key=GEMINI_API_KEY)
with open(BASE / "vlm_system_prompt.md") as f:
    SYSTEM_PROMPT = f.read()

def call_gemini(png_bytes, delay=15):
    time.sleep(delay)
    for m in ["gemini-2.5-flash"]:
        try:
            resp = client.models.generate_content(
                model=m,
                contents=[
                    SYSTEM_PROMPT,
                    types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
                ],
            )
            return resp.text.strip(), m
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                continue
            raise
    return "ERROR", "none"

def parse(raw, seg_start, seg_dur=30):
    noise_type = confidence = "unknown"
    t_start_pct = 0.0; t_end_pct = 100.0
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("NOISE_TYPE:"):
            noise_type = line.split(":",1)[1].strip()
        elif line.startswith("CONFIDENCE:"):
            confidence = line.split(":",1)[1].strip()
        elif line.startswith("TIME_START:"):
            try: t_start_pct = float(line.split(":",1)[1].strip().rstrip("%"))
            except: pass
        elif line.startswith("TIME_END:"):
            try: t_end_pct   = float(line.split(":",1)[1].strip().rstrip("%"))
            except: pass
    abs_start = round(seg_start + seg_dur * t_start_pct / 100, 1)
    abs_end   = round(seg_start + seg_dur * t_end_pct   / 100, 1)
    return noise_type, confidence, abs_start, abs_end


# ── Test cases ────────────────────────────────────────────────────────────────
# Each: (wav_file, segment_start, noise_abs_start, noise_abs_end, noise_src, true_type)
wav1 = GENUINE / "rasa_Hindi_Male_003.wav"
wav2 = GENUINE / "rasa_Hindi_Male_007.wav"
wav3 = GENUINE / "rasa_Hindi_Male_015.wav"

test_cases = [
    # file,  seg_start, noise_start, noise_end, noise_src,        true_type,     label
    (wav1,    0,         0,           30,        rain_audio,       "rain",        "START seg, RAIN full 30s"),
    (wav1,    60,        60,          90,        crowd_audio,      "crowd",       "MID   seg, CROWD full 30s"),
    (wav2,    0,         0,           30,        outdoor_audio,    "outdoor",     "START seg, OUTDOOR full 30s"),
    (wav2,    60,        60,          90,        hvac_audio,       "hvac",        "MID   seg, HVAC full 30s"),
    (wav3,    0,         0,           30,        whitenoise_audio, "white_noise", "START seg, WHITE_NOISE full 30s"),
    (wav3,    90,        90,          120,       human_audio,      "human",       "END   seg, HUMAN full 30s"),
]

print(f"\n{'─'*100}")
print(f"  {'Case':<35} {'TrueType':<13} {'DetectedType':<14} {'Conf':<8} "
      f"{'TrueRange':<14} {'DetectedRange':<14} {'Match'}")
print(f"{'─'*100}")

correct = 0
for wav, seg_start, ns, ne, noise_src, true_type, label in test_cases:
    audio, _ = librosa.load(str(wav), sr=SR, mono=True)
    if len(audio) < 120 * SR:
        audio = np.pad(audio, (0, 120*SR - len(audio)))

    # Mix noise into sub-window
    mixed   = mix_partial(audio, seg_start, ns, ne, noise_src)

    # Extract 30s manipulated segment
    seg     = mixed[int(seg_start*SR) : int((seg_start+30)*SR)]

    # Baseline: genuine (unmanipulated) version of the same segment
    baseline = audio[int(seg_start*SR) : int((seg_start+30)*SR)]

    # Use raw spectrogram — Gemini understands real spectrograms, not diff images
    png     = make_raw_spec(seg)

    # Send to Gemini
    raw, model_used = call_gemini(png)
    det_type, conf, det_start, det_end = parse(raw, seg_start)

    match   = "✓" if det_type.lower() == true_type else "✗"
    if det_type.lower() == true_type:
        correct += 1

    true_range = f"{ns}s–{ne}s"
    det_range  = f"{det_start}s–{det_end}s"
    print(f"  {label:<35} {true_type:<13} {det_type:<14} {conf:<8} "
          f"{true_range:<14} {det_range:<14} {match}")

print(f"{'─'*100}")
print(f"  Noise type accuracy: {correct}/{len(test_cases)} "
      f"({correct/len(test_cases)*100:.0f}%)")
