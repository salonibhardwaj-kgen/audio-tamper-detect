"""
Noise floor stationarity analyzer using Itakura-Saito divergence.

A genuine recording has a stationary noise floor — all silent regions share
the same spectral profile. Selective denoising or noise addition breaks
stationarity: IS divergence between treated and untreated segments jumps
from the genuine-recording baseline (~< 0.15) to 0.5–2.0+.

Reference: Févotte et al., "Nonnegative Matrix Factorization with the
Itakura-Saito Divergence" (NMF-IS divergence formulation).
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Optional

# ── Analysis parameters ─────────────────────────────────────────────────────
SEGMENT_DURATION  = 2.0    # seconds per analysis segment
FRAME_LENGTH      = 1024   # FFT/framing size (~46 ms @ 22050 Hz)
HOP_LENGTH        = 256    # hop between frames (~12 ms)
SILENCE_PERCENTILE = 25    # bottom N% of frames by RMS are treated as silence
MIN_SILENT_FRAMES  = 8     # minimum silent frames needed to estimate noise floor
# Absolute IS threshold — kept as fallback when fewer than 3 valid segments exist.
# Calibrated on LibriSpeech + ESC-50 studio recordings; not suitable as the primary
# decision criterion for field/conversational recordings.
IS_THRESHOLD       = 0.90
# Calibrated on real audio (LibriSpeech + ESC-50): genuine SFM_dev max = 0.2007.
# Raised to 0.35 after CORAAL field recording testing — natural SFM variability
# in unprocessed field recordings reaches 0.35 due to speech content leaking
# into some segment noise floor estimates.  Denoising must whiten beyond this
# to be detectable via the SFM check.
SFM_THRESHOLD      = 0.35
# Relative threshold: flag if row_ratio (max_row_mean / median_row_mean) > RELATIVE_K.
# Genuine CORAAL field recordings peak at row_ratio ~12 (natural acoustic variation).
# A spliced or denoised segment is an outlier against every other segment, so its
# row_ratio is expected >> 15.  Calibrate once manipulated test cases are available.
RELATIVE_K         = 15.0
# Energy drop threshold: median_energy / min_energy across segments.
# Genuine Rasa ceiling = 164.  Missed denoised clips start at 11,959.
# Threshold 500 sits cleanly between both with a 3x margin on each side.
ENERGY_DROP_THRESHOLD = 500.0


# ── Core math ────────────────────────────────────────────────────────────────

def _is_divergence(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Itakura-Saito divergence from spectral distribution P to Q.
    d_IS(P, Q) = mean_k [ P_k/Q_k - log(P_k/Q_k) - 1 ]
    Always >= 0; equals 0 iff P == Q.
    Using mean-per-bin so the value is independent of FFT size.
    """
    P = np.maximum(P, 1e-12)
    Q = np.maximum(Q, 1e-12)
    ratio = P / Q
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def symmetric_is_divergence(P: np.ndarray, Q: np.ndarray) -> float:
    """Symmetric IS divergence: average of d_IS(P,Q) and d_IS(Q,P)."""
    return (_is_divergence(P, Q) + _is_divergence(Q, P)) / 2.0


# ── Spectral flatness ────────────────────────────────────────────────────────

def compute_sfm(noise_floor: np.ndarray) -> float:
    """
    Spectral Flatness Measure: geometric_mean / arithmetic_mean.
    Range [0, 1]; 1 = white noise (flat spectrum), near 0 = tonal/peaked.
    Denoising whitens the residual noise, raising SFM on treated segments.
    """
    P = np.maximum(noise_floor, 1e-12)
    return float(np.exp(np.mean(np.log(P))) / np.mean(P))


# ── Noise floor estimation ────────────────────────────────────────────────────

def estimate_noise_floor(
    segment: np.ndarray,
    frame_length: int = FRAME_LENGTH,
    hop_length: int = HOP_LENGTH,
    silence_percentile: int = SILENCE_PERCENTILE,
) -> Optional[np.ndarray]:
    """
    Estimate noise floor spectrum from a single audio segment using
    per-bin minimum statistics (10th percentile per frequency bin).

    Speech energy at any given frequency bin is intermittent; the 10th
    percentile across all frames captures the persistent noise floor
    regardless of how much speech is in the segment.  This is far more
    robust than selecting "quiet frames" by RMS, which breaks on quiet
    recordings (gate floor collapses the selection window) and on speech-
    dense segments (broadband consonants leak into the estimate).

    Returns None if the segment is too short or the estimated floor is white
    (indicating broadcast processing / noise gate artifacts).
    """
    if len(segment) < frame_length:
        return None, None

    frames = librosa.util.frame(segment, frame_length=frame_length, hop_length=hop_length)
    if frames.shape[1] < MIN_SILENT_FRAMES:
        return None, None

    # Power spectrum for every frame (rfft → positive frequencies only)
    power = np.abs(np.fft.rfft(frames, axis=0)) ** 2  # shape: (bins, n_frames)

    # Per-bin 10th percentile: the noise floor level at each frequency.
    # Speech is transient so its percentile sits above the noise floor;
    # the 10th percentile reliably finds the background regardless of
    # speech density.
    noise_floor = np.percentile(power, 10, axis=1)

    total = float(np.sum(noise_floor))
    if total < 1e-12:
        return None, None

    # L1-normalise → probability distribution (shape, not level)
    noise_floor_norm = (noise_floor / total).astype(np.float64)

    # SFM guard: white floor (SFM > 0.5) means broadcast processing or
    # noise gate artifacts dominate; not analysable.
    if compute_sfm(noise_floor_norm) > 0.5:
        return None, None

    return noise_floor_norm, total


# ── Main API ─────────────────────────────────────────────────────────────────

def _adaptive_threshold(upper: np.ndarray, k: float) -> float:
    """
    Relative threshold: median + k * MAD.
    MAD floor is 10% of the median to prevent collapse on perfectly uniform files.
    """
    median = float(np.median(upper))
    mad    = float(np.median(np.abs(upper - median)))
    mad    = max(mad, median * 0.10)
    return median + k * mad


def _row_outlier_score(div_matrix: np.ndarray) -> tuple[float, int]:
    """
    Structural check: compute each segment's mean divergence to all others
    (its 'row mean'). A manipulated segment is an outlier against EVERY
    other segment, so its row mean will be much higher than the rest.

    Returns (ratio, outlier_idx):
      ratio       — max_row_mean / median_row_mean  (higher = more suspicious)
      outlier_idx — which segment index has the highest row mean
    """
    n = div_matrix.shape[0]
    row_means = np.array([
        np.mean([div_matrix[i, j] for j in range(n) if j != i])
        for i in range(n)
    ])
    median_rm = float(np.median(row_means))
    max_rm    = float(np.max(row_means))
    ratio     = max_rm / max(median_rm, 1e-10)
    outlier   = int(np.argmax(row_means))
    return ratio, outlier


def analyze_audio(
    audio: np.ndarray,
    sr: int,
    segment_duration: float = SEGMENT_DURATION,
    threshold: float = IS_THRESHOLD,
    relative_k: Optional[float] = RELATIVE_K,
    sfm_threshold: float = SFM_THRESHOLD,
    energy_drop_threshold: float = ENERGY_DROP_THRESHOLD,
) -> dict:
    """
    Run noise stationarity analysis on a raw audio array.

    Parameters
    ----------
    threshold   : absolute IS divergence cutoff (used when relative_k is None
                  or there are fewer than 3 valid segments).
    relative_k  : if set, use the adaptive threshold  median + k * MAD
                  instead of the fixed threshold.  Recommended k = 2.0–3.0.
                  Eliminates the need to calibrate a global threshold.

    Returns a dict with:
      n_valid_segments    — number of segments with usable silence
      segment_times       — start time (s) of each valid segment
      noise_floors        — list of normalised power spectra (np.ndarray)
      divergence_matrix   — n×n symmetric IS divergence matrix
      max_divergence      — worst-case pairwise divergence
      mean_divergence     — mean of upper-triangle divergences
      std_divergence      — std of upper-triangle divergences
      median_divergence   — median of upper-triangle divergences
      mad_divergence      — MAD of upper-triangle divergences
      adaptive_threshold  — threshold actually used (absolute or relative)
      threshold_mode      — "relative" or "absolute"
      is_manipulated      — True if max_divergence > adaptive_threshold OR sfm_max_dev > sfm_threshold
      sfm_values          — list of per-segment SFM values
      sfm_max_dev         — max(sfm) - min(sfm) across segments
      sfm_triggered       — True if SFM check alone flagged this file
      verdict             — human-readable string
    """
    segment_samples = int(segment_duration * sr)
    n_segs = len(audio) // segment_samples

    valid_segments: list[tuple[int, np.ndarray, float]] = []
    for i in range(n_segs):
        seg = audio[i * segment_samples : (i + 1) * segment_samples]
        nf, energy = estimate_noise_floor(seg)
        if nf is not None:
            valid_segments.append((i, nf, energy))

    if len(valid_segments) < 2:
        return {
            "n_valid_segments": len(valid_segments),
            "error": "Need at least 2 segments with detectable silence.",
            "is_manipulated": None,
            "verdict": "INCONCLUSIVE — insufficient silent regions",
        }

    # Broadcast / noise-gate detection: if the median SFM of estimated noise
    # floors is above 0.30, the "noise floor" is broadcast compression or gate
    # artifacts, not genuine room noise. IS divergence is meaningless here.
    # Genuine room noise floors (LibriSpeech, NOIZEUS) have median SFM ~0.05–0.20.
    # Broadcast-processed audio (BBC, compressed podcasts) lands at 0.30–0.55.
    floor_sfm_values = [compute_sfm(nf) for _, nf, _ in valid_segments]
    median_floor_sfm = float(np.median(floor_sfm_values))
    if median_floor_sfm > 0.30:
        return {
            "n_valid_segments":  len(valid_segments),
            "median_floor_sfm":  median_floor_sfm,
            "is_manipulated":    None,
            "verdict":           f"UNANALYZABLE — broadcast processing / noise gating detected "
                                 f"(median noise floor SFM={median_floor_sfm:.3f} > 0.30)",
        }

    n = len(valid_segments)
    div_matrix = np.zeros((n, n), dtype=np.float64)

    for a in range(n):
        for b in range(a + 1, n):
            d = symmetric_is_divergence(valid_segments[a][1], valid_segments[b][1])
            div_matrix[a, b] = d
            div_matrix[b, a] = d

    # Energy drop check: median_energy / min_energy.
    # Denoising drives one segment's noise floor energy toward zero while
    # leaving others unchanged → ratio spikes far above the genuine ceiling (164).
    energy_levels = np.array([e for _, _, e in valid_segments])
    energy_median = float(np.median(energy_levels))
    energy_min    = float(np.min(energy_levels))
    energy_drop_ratio = energy_median / max(energy_min, 1e-20)
    energy_triggered  = energy_drop_ratio > energy_drop_threshold

    upper      = div_matrix[np.triu_indices(n, k=1)]
    max_div    = float(np.max(upper))
    mean_div   = float(np.mean(upper))
    std_div    = float(np.std(upper))
    median_div = float(np.median(upper))
    mad_div    = float(np.median(np.abs(upper - median_div)))

    # Row means: average IS of each segment against all others.
    # A manipulated segment is an outlier against EVERY other segment so its
    # row mean is much higher than the rest; naturally variable segments only
    # diverge from SOME others and have a moderate row mean.
    row_means = np.array([
        np.mean([div_matrix[i, j] for j in range(n) if j != i])
        for i in range(n)
    ])
    max_row_mean   = float(np.max(row_means))
    median_row_mean = float(np.median(row_means))
    row_ratio      = max_row_mean / max(median_row_mean, 1e-10)
    outlier_seg    = int(np.argmax(row_means))

    # Choose threshold mode
    if relative_k is not None and n >= 3:
        # Relative mode: flag if row_ratio > relative_k.
        # row_ratio = max_row_mean / median_row_mean measures how much the
        # worst segment stands out against all others.  Genuine recordings
        # have natural variation (row_ratio ~2–12); a spliced or denoised
        # segment is an outlier against every other segment (row_ratio >> 15).
        thr_used = relative_k
        thr_mode = "relative"
        is_triggered = row_ratio > thr_used
    else:
        thr_used = threshold
        thr_mode = "absolute"
        is_triggered = max_div > thr_used

    # Spectral Flatness Measure per segment
    sfm_values  = np.array([compute_sfm(nf) for _, nf, _ in valid_segments])
    sfm_max_dev = float(np.max(sfm_values) - np.min(sfm_values))
    sfm_triggered = sfm_max_dev > sfm_threshold

    # Combined decision: IS divergence OR energy drop.
    manipulated = is_triggered or energy_triggered

    if manipulated:
        triggers = []
        if is_triggered:
            if thr_mode == "relative":
                triggers.append(f"row_ratio={row_ratio:.2f}>{thr_used:.1f}")
            else:
                triggers.append(f"IS={max_div:.4f}>{thr_used:.4f}")
        if energy_triggered:
            triggers.append(f"energy_drop={energy_drop_ratio:.1f}>{energy_drop_threshold:.0f}")
        verdict = f"MANIPULATED  ({', '.join(triggers)}, row_ratio={row_ratio:.2f})"
    else:
        if thr_mode == "relative":
            verdict = (
                f"GENUINE      (row_ratio={row_ratio:.2f}<={thr_used:.1f}, "
                f"energy_drop={energy_drop_ratio:.1f}<={energy_drop_threshold:.0f})"
            )
        else:
            verdict = (
                f"GENUINE      (max IS={max_div:.4f}<={thr_used:.4f}, "
                f"energy_drop={energy_drop_ratio:.1f}<={energy_drop_threshold:.0f}, "
                f"row_ratio={row_ratio:.2f})"
            )

    return {
        "n_valid_segments":   n,
        "segment_times":      [i * segment_duration for i, _, _ in valid_segments],
        "noise_floors":       [nf for _, nf, _ in valid_segments],
        "divergence_matrix":  div_matrix,
        "max_divergence":     max_div,
        "mean_divergence":    mean_div,
        "std_divergence":     std_div,
        "median_divergence":  median_div,
        "mad_divergence":     mad_div,
        "row_outlier_ratio":  row_ratio,
        "outlier_segment":    outlier_seg,
        "adaptive_threshold": thr_used,
        "threshold_mode":     thr_mode,
        "sfm_values":         sfm_values.tolist(),
        "sfm_max_dev":        sfm_max_dev,
        "sfm_triggered":      sfm_triggered,
        "energy_levels":      energy_levels.tolist(),
        "energy_drop_ratio":  energy_drop_ratio,
        "energy_triggered":   energy_triggered,
        "is_manipulated":     manipulated,
        "verdict":            verdict,
    }


def analyze_file(
    path: str,
    segment_duration: float = SEGMENT_DURATION,
    threshold: float = IS_THRESHOLD,
    relative_k: Optional[float] = RELATIVE_K,
    target_sr: Optional[int] = None,
    sfm_threshold: float = SFM_THRESHOLD,
    energy_drop_threshold: float = ENERGY_DROP_THRESHOLD,
) -> dict:
    """Load a WAV/FLAC/etc. file and run analyze_audio."""
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    result = analyze_audio(audio, sr, segment_duration=segment_duration,
                           threshold=threshold, relative_k=relative_k,
                           sfm_threshold=sfm_threshold,
                           energy_drop_threshold=energy_drop_threshold)
    result["path"]       = path
    result["sr"]         = sr
    result["duration_s"] = len(audio) / sr
    return result


# ── Quick CLI usage ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json

    if len(sys.argv) < 2:
        print("Usage: python noise_analyzer.py <audio_file> [threshold]")
        sys.exit(1)

    thr = float(sys.argv[2]) if len(sys.argv) > 2 else IS_THRESHOLD
    res = analyze_file(sys.argv[1], threshold=thr)

    print(f"\nFile     : {res['path']}")
    print(f"Duration : {res.get('duration_s', '?'):.1f}s   SR: {res.get('sr', '?')} Hz")
    print(f"Segments : {res['n_valid_segments']} valid")
    print(f"IS max   : {res.get('max_divergence', 'N/A')}")
    print(f"IS mean  : {res.get('mean_divergence', 'N/A')}")
    print(f"Verdict  : {res['verdict']}")

    if "divergence_matrix" in res:
        print("\nPairwise IS divergence matrix:")
        mat = res["divergence_matrix"]
        times = res["segment_times"]
        header = "      " + "  ".join(f"{t:5.1f}s" for t in times)
        print(header)
        for i, t in enumerate(times):
            row = "  ".join(f"{mat[i,j]:6.4f}" for j in range(len(times)))
            print(f"{t:5.1f}s  {row}")
