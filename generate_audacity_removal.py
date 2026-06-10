"""
Generate Audacity-style noise removal training data.

Simulates Audacity's spectral subtraction algorithm (Boll 1979) with
default Audacity parameters:
  - Noise Reduction: 6 dB  (moderate, not aggressive like noisereduce)
  - Sensitivity:     6
  - Frequency smoothing applied across neighbouring bins

Applies removal to one 30s segment per clip in 4 variants:
  start  : 0–30s
  mid    : 45–75s
  end    : 90–120s
  random : random 30s window

Output:
  datasets/rasa_manipulated/audacity_removal/   400 WAVs
  datasets/spectrograms_audacity_removal/        400 PNGs
"""

import numpy as np
import librosa
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import random

BASE       = Path(os.environ.get("NOISE_BASE_DIR", str(Path(__file__).parent)))
RASA_DIR   = BASE / "datasets" / "rasa"
OUT_WAV    = BASE / "datasets" / "rasa_manipulated" / "audacity_removal"
OUT_SPEC   = BASE / "datasets" / "spectrograms_audacity_removal"

OUT_WAV.mkdir(parents=True, exist_ok=True)
OUT_SPEC.mkdir(parents=True, exist_ok=True)

SR         = 22050
CLIP_LEN   = 120          # seconds
SEG        = 30           # segment length
N_FFT      = 2048
HOP        = 512
N_MELS     = 128
FMAX       = 8000
NOISE_DB   = 6            # Audacity default — moderate reduction
SMOOTH_BANDS = 6          # Audacity default frequency smoothing bands

# ── Audacity-style spectral subtraction ────────────────────────────────────
def audacity_reduce(audio, sr, seg_start, seg_end,
                    noise_db=NOISE_DB, smooth_bands=SMOOTH_BANDS):
    """
    Spectral subtraction matching Audacity's default parameters.
      - Noise profile: first 1s of clip (Audacity requires a quiet section)
      - Reduction:     noise_db dB subtraction (linear factor alpha)
      - Smoothing:     across smooth_bands neighbouring frequency bins
      - Floor:         10% of original magnitude (prevents complete silence)
    """
    S     = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP)
    mag   = np.abs(S)
    phase = np.angle(S)

    # Noise profile from first 1s
    profile_frames = max(1, int(1.0 * sr / HOP))
    noise_profile  = mag[:, :profile_frames].mean(axis=1, keepdims=True)

    # Frequency smoothing on noise profile (Audacity smoothing bands)
    if smooth_bands > 0:
        kernel = np.ones(smooth_bands * 2 + 1) / (smooth_bands * 2 + 1)
        smoothed = np.convolve(noise_profile[:, 0], kernel, mode="same")
        noise_profile = smoothed[:, np.newaxis]

    # Linear subtraction factor for noise_db
    alpha = 10 ** (noise_db / 20.0)   # 6 dB → alpha ≈ 2.0

    # Frame range for the manipulated segment
    f_start = int(seg_start * sr / HOP)
    f_end   = min(int(seg_end * sr / HOP), mag.shape[1])

    # Apply subtraction — floor at 10% of original (Audacity keeps residue)
    seg = mag[:, f_start:f_end].copy()
    reduced = seg - alpha * noise_profile
    mag[:, f_start:f_end] = np.maximum(reduced, 0.10 * seg)

    # Reconstruct
    S_out = mag * np.exp(1j * phase)
    return librosa.istft(S_out, hop_length=HOP, length=len(audio))


# ── 2-panel spectrogram ─────────────────────────────────────────────────────
def save_spectrogram(audio, sr, out_path):
    S    = librosa.feature.melspectrogram(y=audio, sr=sr,
                                          n_mels=N_MELS, fmax=FMAX)
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("black")

    # Panel 1 — full spectrum
    axes[0].imshow(S_db, aspect="auto", origin="lower",
                   cmap="magma", vmin=-80, vmax=0)
    axes[0].axis("off")

    # Panel 2 — noise floor zoom (bottom 20 mel bins, boosted contrast)
    floor = S_db[:20, :]
    axes[1].imshow(floor, aspect="auto", origin="lower",
                   cmap="inferno",
                   vmin=np.percentile(floor, 5),
                   vmax=np.percentile(floor, 95))
    axes[1].axis("off")

    plt.tight_layout(pad=0)
    fig.savefig(str(out_path), dpi=100, bbox_inches="tight",
                facecolor="black")
    plt.close()


# ── Segment definitions ─────────────────────────────────────────────────────
def get_segments(clip_len=CLIP_LEN, seg=SEG):
    rand_start = random.randint(0, clip_len - seg)
    return {
        "start":  (0,               seg),
        "mid":    (45,              45 + seg),
        "end":    (clip_len - seg,  clip_len),
        "random": (rand_start,      rand_start + seg),
    }


# ── Main ────────────────────────────────────────────────────────────────────
wav_files = sorted(RASA_DIR.glob("rasa_Hindi_Male_*.wav"))
print(f"Found {len(wav_files)} genuine clips")
print(f"Generating {len(wav_files) * 4} Audacity-style removal clips...\n")

total = 0
errors = 0

for wav in wav_files:
    try:
        audio, sr = librosa.load(str(wav), sr=SR, mono=True, duration=CLIP_LEN)
        if len(audio) < CLIP_LEN * SR:
            audio = np.pad(audio, (0, CLIP_LEN * SR - len(audio)))

        segments = get_segments()

        for variant, (seg_start, seg_end) in segments.items():
            stem    = wav.stem                          # rasa_Hindi_Male_001
            out_name = f"{stem}_audacity_{variant}"

            # Generate manipulated audio
            manipulated = audacity_reduce(audio, sr, seg_start, seg_end)

            # Save WAV
            wav_out = OUT_WAV / f"{out_name}.wav"
            sf.write(str(wav_out), manipulated, sr)

            # Save spectrogram
            spec_out = OUT_SPEC / f"{out_name}.png"
            save_spectrogram(manipulated, sr, spec_out)

            total += 1
            if total % 40 == 0:
                print(f"  {total} / {len(wav_files) * 4} done...")

    except Exception as e:
        print(f"  ERROR {wav.name}: {e}")
        errors += 1

print(f"\nDone — {total} clips generated, {errors} errors")
print(f"WAVs  → {OUT_WAV}")
print(f"Specs → {OUT_SPEC}")
