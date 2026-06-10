"""
Generate mel spectrogram images from audio clips.
Saves PNGs to datasets/spectrograms/
"""

import sys, numpy as np, librosa, librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE      = Path(os.environ.get("NOISE_BASE_DIR", str(Path(__file__).parent)))
RASA_DIR  = BASE / "datasets" / "rasa"
MANIP_DIR = BASE / "datasets" / "rasa_manipulated" / "noise_removal"
OUT_DIR   = BASE / "datasets" / "spectrograms"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 22050
N_MELS    = 128
FMAX      = 8000


def generate_spectrogram(audio_path: Path, out_path: Path, label: str = ""):
    audio, sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
    audio     = audio[:120 * TARGET_SR]

    S    = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, fmax=FMAX)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Two panels: full spectrogram + noise floor zoom (bottom 20 mel bins)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), dpi=120,
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Panel 1 — full spectrogram
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel",
                             fmax=FMAX, ax=ax1, cmap="magma")
    for t in [30, 60, 90]:
        ax1.axvline(x=t, color="cyan", linewidth=1.0, linestyle="--", alpha=0.8)
        ax1.text(t + 0.5, 7000, f"{t}s", color="cyan", fontsize=7)
    ax1.set_xlabel("")
    ax1.set_ylabel("Frequency (mel)")

    # Panel 2 — noise floor zoom: bottom 20 mel bins, boosted contrast
    noise_floor_db = S_db[:20, :]
    vmin = np.percentile(noise_floor_db, 5)
    vmax = np.percentile(noise_floor_db, 95)
    librosa.display.specshow(noise_floor_db, sr=sr, x_axis="time",
                             ax=ax2, cmap="inferno", vmin=vmin, vmax=vmax)
    for t in [30, 60, 90]:
        ax2.axvline(x=t, color="cyan", linewidth=1.0, linestyle="--", alpha=0.8)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Low freq")

    plt.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close()

    size_kb = out_path.stat().st_size // 1024
    print(f"  Saved: {out_path.name}  ({size_kb} KB)")
    return out_path


# ── Genuine clips — all 99 ────────────────────────────────────────────────────
print("Generating genuine spectrograms (all 99)...")
genuine_clips = sorted(RASA_DIR.glob("rasa_Hindi_Male_*.wav"))
for clip in genuine_clips:
    out = OUT_DIR / f"genuine_{clip.stem}.png"
    if out.exists():
        continue
    generate_spectrogram(clip, out, label="GENUINE")

# ── Manipulated clips — all 99 ────────────────────────────────────────────────
print("\nGenerating manipulated spectrograms (all 99)...")
manip_clips = sorted(MANIP_DIR.glob("*.wav"))
for clip in manip_clips:
    out = OUT_DIR / f"manipulated_{clip.stem}.png"
    if out.exists():
        continue
    generate_spectrogram(clip, out, label="MANIPULATED")

print(f"\nDone. All spectrograms saved to:\n  {OUT_DIR}")
genuine_count = len(list(OUT_DIR.glob("genuine_*.png")))
manip_count   = len(list(OUT_DIR.glob("manipulated_*.png")))
print(f"  Genuine     : {genuine_count}")
print(f"  Manipulated : {manip_count}")
print(f"  Total       : {genuine_count + manip_count}")
