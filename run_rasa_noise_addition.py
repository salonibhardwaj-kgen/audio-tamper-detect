"""
Rasa Hindi Male — multi-segment noise addition.

For each genuine 120s clip, splits into 4 × 30s segments and randomly
assigns different noise types to 1–3 of those segments. Each noisy segment
gets a distinct noise type. Segment assignments are encoded in the filename
for ground-truth evaluation.

Filename format (4-code suffix, one code per segment):
  rasa_Hindi_Male_042_multi_cr_no_hv_wn.wav
  → seg0=crowd, seg1=genuine, seg2=hvac, seg3=white_noise

Type codes: hv=hvac  wn=white_noise  cr=crowd  ra=rain  ou=outdoor  hu=human  no=genuine

25 clips × 4 variants = 100 manipulated WAVs
Output → datasets/rasa_manipulated/noise_addition/
"""

import random, numpy as np, soundfile as sf, librosa
from pathlib import Path

BASE        = Path(os.environ.get("NOISE_BASE_DIR", str(Path(__file__).parent)))
GENUINE_DIR = BASE / "datasets" / "rasa"
NOISE_DIR   = BASE / "datasets" / "noise_sources"
OUT_DIR     = BASE / "datasets" / "rasa_manipulated" / "noise_addition"

TARGET_SR         = 22050
TOTAL_DUR         = 120
SEG_DUR           = 30
MIX_RATIO         = 0.60
CLIPS_PER_TYPE    = 100
VARIANTS_PER_CLIP = 4

random.seed(42)

TYPE_CODE = {
    "hvac":       "hv",
    "white_noise": "wn",
    "crowd":      "cr",
    "rain":       "ra",
    "outdoor":    "ou",
    "human":      "hu",
}
CODE_TYPE = {v: k for k, v in TYPE_CODE.items()}

NOISE_SOURCES = {
    "hvac":        [NOISE_DIR / "hvac" / "hvac_exhaust_fan.wav",
                    NOISE_DIR / "hvac" / "hvac_ventilation.wav"],
    "white_noise": [NOISE_DIR / "white_noise" / "white_noise.wav"],
    "crowd":       [NOISE_DIR / "crowd" / "crowd_murmur.wav",
                    NOISE_DIR / "crowd" / "crowd_congenial.wav",
                    NOISE_DIR / "crowd" / "busy_city_crowd.wav"],
    "rain":        [NOISE_DIR / "rain" / "rain_combined.wav"],
    "outdoor":     [NOISE_DIR / "outdoor" / "birds_outdoor.wav",
                    NOISE_DIR / "outdoor" / "outdoor_ambience.wav"],
    "human":       [NOISE_DIR / "human" / "office_scene.wav"],
}

# ── Load and cache noise sources ───────────────────────────────────────────────
print("Loading noise sources...")
noise_cache = {}
for ntype, paths in NOISE_SOURCES.items():
    segments = []
    for p in paths:
        if not p.exists():
            print(f"  WARNING: {p} not found — skipping")
            continue
        audio, sr = librosa.load(str(p), sr=TARGET_SR, mono=True)
        segments.append(audio)
    if not segments:
        print(f"  SKIP {ntype} — no files found")
        continue
    combined = np.concatenate(segments)
    combined = combined / (np.max(np.abs(combined)) + 1e-9)
    noise_cache[ntype] = combined
    print(f"  {ntype:<12} : {len(combined)/TARGET_SR:.1f}s loaded")

noise_types = list(noise_cache.keys())


def get_noise_segment(ntype: str, n_samples: int) -> np.ndarray:
    noise = noise_cache[ntype]
    if len(noise) >= n_samples:
        start = random.randint(0, len(noise) - n_samples)
        return noise[start: start + n_samples]
    repeats = (n_samples // len(noise)) + 2
    return np.tile(noise, repeats)[:n_samples]


# ── Get genuine clips ──────────────────────────────────────────────────────────
all_clips = sorted(GENUINE_DIR.glob("rasa_Hindi_Male_*.wav"))
print(f"\nFound {len(all_clips)} genuine clips")
clips_copy = list(all_clips)
random.shuffle(clips_copy)
selected_clips = clips_copy[:CLIPS_PER_TYPE]
print(f"Selected {len(selected_clips)} clips, {VARIANTS_PER_CLIP} variants each")
print(f"Target: {len(selected_clips) * VARIANTS_PER_CLIP} manipulated WAVs\n")

OUT_DIR.mkdir(parents=True, exist_ok=True)

seg_samples = SEG_DUR * TARGET_SR
n_segs      = TOTAL_DUR // SEG_DUR   # 4

total_created = 0
total_skipped = 0

for clip_path in selected_clips:
    audio, sr = librosa.load(str(clip_path), sr=TARGET_SR, mono=True)
    if len(audio) < TOTAL_DUR * TARGET_SR:
        print(f"  SKIP {clip_path.name} — too short")
        continue
    audio = audio[:TOTAL_DUR * TARGET_SR]

    # Generate VARIANTS_PER_CLIP different noise assignments for this clip
    for v in range(VARIANTS_PER_CLIP):
        # Randomly choose how many segments to corrupt (1–3)
        n_noisy = random.randint(1, 3)
        noisy_seg_idxs = sorted(random.sample(range(n_segs), n_noisy))
        # Assign a distinct noise type to each noisy segment
        assigned_types = random.sample(noise_types, n_noisy)
        seg_assignment = dict(zip(noisy_seg_idxs, assigned_types))  # {idx: type}

        # Build 4-code filename suffix (one code per segment, left to right)
        codes = [TYPE_CODE.get(seg_assignment.get(i, "none"), "no") for i in range(n_segs)]
        suffix = "_".join(codes)
        out_name = f"{clip_path.stem}_multi_{suffix}.wav"
        out_path = OUT_DIR / out_name

        if out_path.exists():
            total_skipped += 1
            continue

        manipulated = audio.copy()
        for seg_idx, ntype in seg_assignment.items():
            s_start = seg_idx * seg_samples
            s_end   = s_start + seg_samples
            speech_seg   = manipulated[s_start:s_end]
            noise_seg    = get_noise_segment(ntype, seg_samples)
            speech_rms   = np.sqrt(np.mean(speech_seg ** 2)) + 1e-9
            noise_rms    = np.sqrt(np.mean(noise_seg  ** 2)) + 1e-9
            noise_scaled = noise_seg * (speech_rms / noise_rms) * MIX_RATIO
            mixed = np.clip(speech_seg + noise_scaled, -1.0, 1.0)
            manipulated[s_start:s_end] = mixed

        sf.write(str(out_path), manipulated, TARGET_SR, subtype="PCM_16")
        total_created += 1
        assignment_str = ", ".join(f"seg{i}={t}" for i, t in sorted(seg_assignment.items()))
        print(f"  {out_name}  [{assignment_str}]")

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Created : {total_created} new WAVs")
print(f"Skipped : {total_skipped} already existed")
print(f"Total   : {len(list(OUT_DIR.glob('*.wav')))} WAVs in {OUT_DIR.name}/")
print(f"{'='*60}")
print(f"\nFilename legend:  hv=hvac  wn=white_noise  cr=crowd")
print(f"                  ra=rain  ou=outdoor  hu=human  no=genuine")
print(f"Example: rasa_..._multi_cr_no_hv_wn.wav")
print(f"         → seg0=crowd  seg1=genuine  seg2=hvac  seg3=white_noise")
