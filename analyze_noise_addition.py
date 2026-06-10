"""
Multi-segment noise addition analyzer.

For each 120s audio (4 × 30s segments):
  1. Compute linear-power noise floor per segment (bottom 40 mel bins, 10th pct)
     — wider band than bottom-20 so crowd (mid) and birds (high) are also detected
  2. Row ratio = segment_energy / baseline_energy
     — baseline = average of the 2 lowest-energy segments (robust against 1 silent seg)
  3. If row_ratio > 2.0 → segment is noisy
  4. Linear-power differential fingerprint:
       diff = max(seg_lin_mean - clean_lin_mean, 0)  (noise contribution, non-negative)
     Compare normalized diff to reference fingerprints (also in linear power)
  5. Shape discriminators on linear diff to break ties between spectrally similar types
  6. Dominant type = type with highest average confidence across noisy segments

Filename GT (4-code suffix):
  rasa_Hindi_Male_042_multi_cr_no_hv_wn.wav
  → seg0=crowd  seg1=genuine  seg2=hvac  seg3=white_noise
  Codes: hv=hvac  wn=white_noise  cr=crowd  ra=rain  ou=outdoor  hu=human  no=genuine

Output → results/rasa_noise_addition.csv
"""

import numpy as np, librosa, csv
from pathlib import Path
from scipy.special import softmax
from scipy.spatial.distance import cosine


def compute_sfm(power_spectrum: np.ndarray) -> float:
    """Spectral Flatness Measure: 0=tonal/concentrated, 1=flat/white-noise-like."""
    ps = power_spectrum + 1e-20
    geometric_mean = np.exp(np.mean(np.log(ps)))
    arithmetic_mean = np.mean(ps)
    return float(geometric_mean / (arithmetic_mean + 1e-20))

BASE        = Path(os.environ.get("NOISE_BASE_DIR", str(Path(__file__).parent)))
NOISE_DIR   = BASE / "datasets" / "noise_sources"
MANIP_DIR   = BASE / "datasets" / "rasa_manipulated" / "noise_addition"
OUT_CSV     = BASE / "results" / "rasa_noise_addition.csv"
OUT_CSV.parent.mkdir(exist_ok=True)

TARGET_SR           = 22050
TOTAL_DUR           = 120
SEG_DUR             = 30
N_MELS              = 128
FMAX                = 8000
ROW_RATIO_THRESHOLD = 1.85
NOISE_FLOOR_BINS    = 40    # wider band: captures crowd (mid) and birds (high) too

CODE_TYPE = {
    "hv": "hvac",
    "wn": "white_noise",
    "cr": "crowd",
    "ra": "rain",
    "ou": "outdoor",
    "hu": "human",
    "no": "none",
}

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


# ── Build reference fingerprints in linear power ───────────────────────────────
def mel_linear(audio: np.ndarray, sr: int) -> np.ndarray:
    """Mean linear-power mel spectrogram per bin → shape (N_MELS,)."""
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, fmax=FMAX)
    return np.mean(S, axis=1)


print("Building reference noise fingerprints (linear power)...")
reference_lin = {}
for ntype, paths in NOISE_SOURCES.items():
    segs = []
    for p in paths:
        if not p.exists():
            continue
        audio, sr = librosa.load(str(p), sr=TARGET_SR, mono=True)
        segs.append(audio[:SEG_DUR * TARGET_SR])
    if not segs:
        continue
    combined = np.concatenate(segs)[:SEG_DUR * TARGET_SR]
    reference_lin[ntype] = mel_linear(combined, TARGET_SR)
    print(f"  {ntype:<12} fingerprint built")


# ── Noise type identification ─────────────────────────────────────────────────
def identify_noise_type(seg_lin: np.ndarray, clean_lin: np.ndarray) -> dict:
    """Ratio-based differential → {noise_type: confidence}.

    Uses seg_lin / clean_lin instead of subtraction so that noise energy in
    speech-quiet bins (very low and very high) is amplified relative to
    speech variation in mid bins.
    """
    diff = np.maximum(seg_lin - clean_lin, 0.0)

    if np.sum(diff) < 1e-20:
        n = len(reference_lin)
        return {k: round(1.0 / n, 4) for k in reference_lin}

    # Ratio excess: amplifies noise where speech is quiet (low/high bins)
    clean_floor = np.max(clean_lin) * 0.001          # prevent /0 on silent bins
    ratio       = seg_lin / (np.maximum(clean_lin, clean_floor) + 1e-30)
    re          = np.maximum(ratio - 1.0, 0.0)       # excess above clean

    re_low  = float(np.sum(re[:13]))    # 0–250 Hz  (speech-free low)
    re_high = float(np.sum(re[90:]))    # 5–8 kHz   (speech-free high)
    re_mid  = float(np.sum(re[13:75]))  # 300–2500 Hz (speech range)
    total_re = re_low + re_high + re_mid + 1e-20

    low_frac  = re_low  / total_re
    high_frac = re_high / total_re
    mid_frac  = re_mid  / total_re

    # SFM on ratio-excess in high bins:
    #   flat (sfm ≈ 1) = white noise / rain broadband
    #   tonal (sfm ≈ 0) = birds (outdoor) / office hum (human)
    sfm_high = compute_sfm(re[90:] + 1e-30)

    # Cosine on low sub-band (HVAC/rain) + mid sub-band (crowd vs hvac)
    # Mid cosine breaks the crowd↔hvac tie: crowd ref has strong mid, hvac ref has weak mid
    low_sub = diff[:13]
    mid_sub = diff[13:75]
    raw_scores = {}
    for ntype, ref_lin in reference_lin.items():
        ref_low = ref_lin[:13]
        ref_mid = ref_lin[13:75]
        denom_low = np.linalg.norm(low_sub) * np.linalg.norm(ref_low)
        denom_mid = np.linalg.norm(mid_sub) * np.linalg.norm(ref_mid)
        sim_low = float(np.dot(low_sub, ref_low) / denom_low) if denom_low > 1e-20 else 0.0
        sim_mid = float(np.dot(mid_sub, ref_mid) / denom_mid) if denom_mid > 1e-20 else 0.0
        # Low sub drives HVAC/rain; mid sub helps separate crowd from HVAC
        raw_scores[ntype] = max(0.6 * sim_low + 0.4 * sim_mid, 0.0)

    # ── Shape discriminators (ratio-based features) ───────────────────────────

    # Crowd: mid-band overwhelmingly dominant (speech chatter 300–2500 Hz)
    if "crowd" in raw_scores:
        if mid_frac > 0.75:
            raw_scores["crowd"] *= 5.0
        else:
            raw_scores["crowd"] *= 0.2

    # HVAC: low-frequency dominant (fan hum < 250 Hz)
    if "hvac" in raw_scores:
        if low_frac > 0.5:
            raw_scores["hvac"] *= 5.0
        elif low_frac > 0.25:
            raw_scores["hvac"] *= 2.0
        else:
            raw_scores["hvac"] *= 0.2

    # White noise: flat high-freq (sfm_high > 0.7) AND high_frac dominant
    if "white_noise" in raw_scores:
        if sfm_high > 0.7 and high_frac > 0.5:
            raw_scores["white_noise"] *= 5.0
        elif sfm_high > 0.5 and low_frac < 0.2:
            raw_scores["white_noise"] *= 2.0
        else:
            raw_scores["white_noise"] *= 0.2

    # Outdoor/birds: tonal high-freq (high_frac > 0.3, sfm_high near 0)
    # Distinguished from human by cosine similarity in high sub-band
    if "outdoor" in raw_scores:
        if high_frac > 0.3 and sfm_high < 0.1:
            raw_scores["outdoor"] *= 5.0
        else:
            raw_scores["outdoor"] *= 0.2

    # Human/office: tonal high-freq but with higher high_frac than outdoor
    # Cosine on high sub-band vs reference fingerprint separates them
    if "human" in raw_scores:
        if high_frac > 0.5 and sfm_high < 0.05:
            raw_scores["human"] *= 5.0
        elif high_frac > 0.3 and sfm_high < 0.1:
            raw_scores["human"] *= 2.0
        else:
            raw_scores["human"] *= 0.2

    # Rain: broadband with significant low content (low_frac > 0.2) + flat high
    if "rain" in raw_scores:
        if low_frac > 0.2 and sfm_high > 0.5:
            raw_scores["rain"] *= 5.0
        elif low_frac > 0.15:
            raw_scores["rain"] *= 2.0
        else:
            raw_scores["rain"] *= 0.2

    names  = list(raw_scores.keys())
    values = np.array([raw_scores[n] for n in names])
    probs  = softmax(values * 20)
    return {n: round(float(p), 4) for n, p in zip(names, probs)}


# ── Parse ground-truth from filename ──────────────────────────────────────────
def parse_ground_truth(filename: str) -> dict:
    """Returns {seg_idx: noise_type} for noisy segments (omits 'none')."""
    name = filename.replace(".wav", "")
    if "_multi_" not in name:
        return {}
    suffix = name.split("_multi_")[1]
    codes  = suffix.split("_")
    return {i: CODE_TYPE[c] for i, c in enumerate(codes)
            if c in CODE_TYPE and CODE_TYPE[c] != "none"}


# ── Core analysis ──────────────────────────────────────────────────────────────
def analyze_audio(audio_path: Path) -> dict:
    audio, sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
    if len(audio) < TOTAL_DUR * TARGET_SR:
        return {"error": "too short"}
    audio = audio[:TOTAL_DUR * TARGET_SR]

    n_segs      = TOTAL_DUR // SEG_DUR   # 4
    seg_samples = SEG_DUR * TARGET_SR

    seg_energies = []
    seg_lin_fps  = []
    for i in range(n_segs):
        seg = audio[i * seg_samples: (i + 1) * seg_samples]
        S   = librosa.feature.melspectrogram(y=seg, sr=sr, n_mels=N_MELS, fmax=FMAX)
        # Detection energy: 10th pct of bottom NOISE_FLOOR_BINS (linear power)
        seg_energies.append(float(np.percentile(S[:NOISE_FLOOR_BINS, :], 10)))
        seg_lin_fps.append(np.mean(S, axis=1))

    seg_energies = np.array(seg_energies)

    # Baseline = average of the 2 lowest-energy segments
    sorted_idx    = np.argsort(seg_energies)
    baseline_e    = float(np.mean(seg_energies[sorted_idx[:2]]))
    row_ratios    = seg_energies / (baseline_e + 1e-20)

    noisy_idxs     = [i for i in range(n_segs) if row_ratios[i] > ROW_RATIO_THRESHOLD]
    is_manipulated = len(noisy_idxs) > 0

    # Fingerprint baseline = the single cleanest (minimum-energy) segment
    clean_idx = int(sorted_idx[0])
    clean_lin = seg_lin_fps[clean_idx]

    seg_results   = []
    type_conf_sum = {}

    for i in range(n_segs):
        seg_info = {
            "seg_idx":    i,
            "time":       f"{i*SEG_DUR}–{(i+1)*SEG_DUR}s",
            "row_ratio":  round(float(row_ratios[i]), 4),
            "is_noisy":   i in noisy_idxs,
            "noise_type": "genuine",
            "noise_conf": 0.0,
        }
        if i in noisy_idxs:
            type_confs = identify_noise_type(seg_lin_fps[i], clean_lin)
            best_type  = max(type_confs, key=type_confs.get)
            best_conf  = type_confs[best_type]
            seg_info["noise_type"] = best_type
            seg_info["noise_conf"] = round(best_conf, 4)
            # Weight by log(row_ratio): high-ratio segments are clearly noisy
            # and should dominate dominant_type; borderline segments (possible
            # false positives from natural speech variation) contribute less.
            weight = float(np.log1p(row_ratios[i]))
            for ntype, conf in type_confs.items():
                type_conf_sum[ntype] = type_conf_sum.get(ntype, 0.0) + conf * weight
        seg_results.append(seg_info)

    if type_conf_sum:
        n_noisy       = len(noisy_idxs)
        dominant_type = max(type_conf_sum, key=type_conf_sum.get)
        dominant_conf = round(type_conf_sum[dominant_type] / n_noisy, 4)
    else:
        dominant_type = "none"
        dominant_conf = 0.0

    return {
        "is_manipulated": is_manipulated,
        "n_noisy_segs":   len(noisy_idxs),
        "dominant_type":  dominant_type if is_manipulated else "none",
        "dominant_conf":  dominant_conf if is_manipulated else 0.0,
        "seg_results":    seg_results,
    }


# ── Run on all clips ───────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("Analyzing multi-segment noise addition clips...")
print(f"{'='*80}\n")

clips = sorted(MANIP_DIR.glob("*.wav")) if MANIP_DIR.exists() else []

if not clips:
    print("No clips found. Run run_rasa_noise_addition.py first.")
    exit(1)

rows        = []
total       = 0
dom_correct = 0
seg_correct = 0
seg_total   = 0

for clip in clips:
    r = analyze_audio(clip)
    if "error" in r:
        continue

    gt = parse_ground_truth(clip.name)
    total += 1

    dom_ok = (r["dominant_type"] in gt.values()) if gt else (r["dominant_type"] == "none")
    dom_correct += int(dom_ok)

    mark     = "✓" if dom_ok else "✗"
    noisy_str = ", ".join(
        f"s{s['seg_idx']}={s['noise_type']}({s['noise_conf']:.2f})"
        for s in r["seg_results"] if s["is_noisy"]
    )
    gt_str   = ", ".join(f"s{i}={t}" for i, t in sorted(gt.items())) or "genuine"

    print(f"  {mark}  {clip.name}")
    print(f"       GT: [{gt_str}]  detected: [{noisy_str or 'none'}]"
          f"  dominant={r['dominant_type']} ({r['dominant_conf']:.2f})")

    for seg in r["seg_results"]:
        i = seg["seg_idx"]
        if i in gt:
            seg_total += 1
            if seg["noise_type"] == gt[i]:
                seg_correct += 1

    row = {
        "file":             clip.name,
        "gt_assignment":    "; ".join(f"s{i}={t}" for i, t in sorted(gt.items())),
        "is_manipulated":   r["is_manipulated"],
        "n_noisy_segs":     r["n_noisy_segs"],
        "dominant_type":    r["dominant_type"],
        "dominant_conf":    r["dominant_conf"],
        "dominant_correct": dom_ok,
    }
    for seg in r["seg_results"]:
        i = seg["seg_idx"]
        row[f"seg{i}_ratio"] = seg["row_ratio"]
        row[f"seg{i}_type"]  = seg["noise_type"]
        row[f"seg{i}_conf"]  = seg["noise_conf"]
    rows.append(row)

# ── Save CSV ───────────────────────────────────────────────────────────────────
if rows:
    fields = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"  Total clips analyzed         : {total}")
if total:
    print(f"  Dominant type correct        : {dom_correct}/{total}  ({100*dom_correct/total:.1f}%)")
if seg_total:
    print(f"  Per-segment type accuracy    : {seg_correct}/{seg_total}  ({100*seg_correct/seg_total:.1f}%)")
print(f"  ROW_RATIO_THRESHOLD used     : {ROW_RATIO_THRESHOLD}")
print(f"  CSV → {OUT_CSV}")
print(f"{'='*80}")
