"""
Quick Claude Vision test on one genuine + one manipulated spectrogram.
Run from terminal where ANTHROPIC_API_KEY is exported.
"""

import sys, os, base64, json
import numpy as np
import librosa, librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import anthropic
from pathlib import Path

BASE      = Path(os.environ.get("NOISE_BASE_DIR", str(Path(__file__).parent)))
RASA_DIR  = BASE / "datasets" / "rasa"
MANIP_DIR = BASE / "datasets" / "rasa_manipulated" / "noise_removal"
TARGET_SR = 22050

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env


def make_spectrogram(audio, sr, title, out_path):
    S    = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(14, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel",
                             fmax=8000, ax=ax, cmap="magma")
    ax.set_title(title)
    for t in [30, 60, 90]:
        ax.axvline(x=t, color="cyan", linewidth=0.8, linestyle="--", alpha=0.7)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()


def ask_claude(image_path, clip_label):
    with open(image_path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode()

    prompt = """This is a mel spectrogram of a 120-second audio clip (speech recording).
Cyan dashed lines mark 30-second boundaries at 30s, 60s, 90s.

Look carefully at the noise floor — the faint background energy visible in the low-frequency bands between speech bursts.

Answer:
1. Is the noise floor visually consistent across the full 120 seconds?
2. If inconsistent — which 30s region looks different and what does it resemble?
   Choices: hvac (steady low hum), crowd (irregular mid-freq murmur), white (flat broadband),
            rain (broadband with spikes), human (rhythmic bursts), outdoor (wind/nature), none
3. How confident are you (0.0–1.0)?

Reply ONLY in valid JSON:
{
  "consistent": true,
  "noise_type": "none",
  "region": "none",
  "confidence": 0.0,
  "reasoning": "one sentence"
}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": prompt}
            ]
        }]
    )
    raw = response.content[0].text.strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        import re
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        result = json.loads(m.group()) if m else {"raw": raw}

    print(f"\n  [{clip_label}]")
    print(f"  Consistent  : {result.get('consistent')}")
    print(f"  Noise type  : {result.get('noise_type')}")
    print(f"  Region      : {result.get('region')}")
    print(f"  Confidence  : {result.get('confidence')}")
    print(f"  Reasoning   : {result.get('reasoning')}")
    return result


# ── Test 1: Genuine clip ───────────────────────────────────────────────────────
print("=" * 55)
print("TEST 1 — Genuine clip (expect: consistent=True, noise_type=none)")
print("=" * 55)

genuine_path = sorted(RASA_DIR.glob("rasa_Hindi_Male_*.wav"))[0]
audio, sr    = librosa.load(str(genuine_path), sr=TARGET_SR, mono=True)
audio        = audio[:120 * TARGET_SR]
spec_path    = BASE / "tmp_genuine_spec.png"
make_spectrogram(audio, sr, f"GENUINE — {genuine_path.name}", spec_path)
r1 = ask_claude(spec_path, "GENUINE")
spec_path.unlink()

# ── Test 2: Noise-removal manipulated clip (start) ────────────────────────────
print("\n" + "=" * 55)
print("TEST 2 — Noise removal at 0–30s (expect: consistent=False, region=0-30s)")
print("=" * 55)

manip_clips  = sorted(MANIP_DIR.glob("*_removal_start.wav"))
if manip_clips:
    manip_path = manip_clips[0]
    audio2, sr2 = librosa.load(str(manip_path), sr=TARGET_SR, mono=True)
    audio2      = audio2[:120 * TARGET_SR]
    spec_path2  = BASE / "tmp_manip_spec.png"
    make_spectrogram(audio2, sr2, f"MANIPULATED (noise removed 0-30s) — {manip_path.name}", spec_path2)
    r2 = ask_claude(spec_path2, "MANIPULATED — noise removal start")
    spec_path2.unlink()
else:
    print("  No noise-removal clips found — skipping test 2")

print("\n" + "=" * 55)
print("Claude Vision smoke test complete.")
print("=" * 55)
