"""
Structured test data generator for noise stationarity analysis.
Uses real audio from:
  • LibriSpeech test-clean  — genuine speech base
  • ESC-50                  — environmental noise injection

One-time dataset setup (run before generating):
  cd "/Users/salonibhardwaj/Desktop/Noise "
  mkdir -p datasets
  wget -O datasets/test-clean.tar.gz https://www.openslr.org/resources/12/test-clean.tar.gz
  tar -xzf datasets/test-clean.tar.gz -C datasets/
  wget -O datasets/ESC-50.zip https://github.com/karoldvl/ESC-50/archive/master.zip
  unzip -q datasets/ESC-50.zip -d datasets/

Directory layout after generation:
test_audio/
├── genuine/
├── noise_added/  crowd/ human/ white/ hvac/ outdoor/ rain/
└── noise_removed/ start/ mid/ end/ randomised/
"""

import csv
import json

import librosa
import numpy as np
import noisereduce as nr
import soundfile as sf
from pathlib import Path
from scipy.signal import butter, sosfilt

# ── Paths ──────────────────────────────────────────────────────────────────
SR       = 22050
DURATION = 12.0
BG_AMP   = 0.008   # light uniform background noise amplitude

BASE_DIR        = Path(__file__).parent
OUT_DIR         = BASE_DIR / "test_audio"
DATASETS        = BASE_DIR / "datasets"
LIBRISPEECH_DIR = DATASETS / "LibriSpeech" / "test-clean"
ESC50_DIR       = DATASETS / "ESC-50-master"
ESC50_META      = ESC50_DIR / "meta" / "esc50.csv"
ESC50_AUDIO     = ESC50_DIR / "audio"

# ESC-50 category names to use per noise type
ESC50_MAP = {
    "crowd":    ["clapping", "laughing"],
    # "human" uses LibriSpeech babble instead of ESC-50 (see human_noise())
    "rain":     ["rain", "thunderstorm"],
    "outdoor":  ["wind", "sea_waves", "chirping_birds"],
    "hvac":     ["vacuum_cleaner", "engine", "washing_machine"],
    # Stationary ambient only — used as consistent background in genuine files.
    # Avoid bursty sounds (wind, crickets, sea_waves) — they create non-stationary floors.
    "ambient":  ["washing_machine", "rain", "vacuum_cleaner"],
}


# ── Dataset caches ─────────────────────────────────────────────────────────

_esc50_cache      = None
_libri_cache      = None
_libri_long_cache = None   # files >= DURATION seconds


def _get_esc50() -> dict:
    global _esc50_cache
    if _esc50_cache is not None:
        return _esc50_cache
    if not ESC50_META.exists():
        print("  [WARN] ESC-50 not found — noise generators will use synthetic fallbacks.")
        _esc50_cache = {}
        return _esc50_cache
    files_by_cat: dict[str, list[Path]] = {}
    with open(ESC50_META, newline="") as f:
        for row in csv.DictReader(f):
            fpath = ESC50_AUDIO / row["filename"]
            if fpath.exists():
                files_by_cat.setdefault(row["category"], []).append(fpath)
    _esc50_cache = files_by_cat
    n_total = sum(len(v) for v in files_by_cat.values())
    print(f"  ESC-50: {n_total} files across {len(files_by_cat)} categories")
    return _esc50_cache


def _get_librispeech() -> list:
    global _libri_cache
    if _libri_cache is not None:
        return _libri_cache
    if not LIBRISPEECH_DIR.exists():
        print("  [WARN] LibriSpeech not found — genuine audio will use synthetic fallback.")
        _libri_cache = []
        return _libri_cache
    files = sorted(LIBRISPEECH_DIR.glob("**/*.flac"))
    _libri_cache = files
    print(f"  LibriSpeech: {len(files)} FLAC files")
    return files


def _get_long_libri_files() -> list:
    """Return LibriSpeech files whose duration >= DURATION seconds.
    Scanned once via header-only sf.info (no audio decoded)."""
    global _libri_long_cache
    if _libri_long_cache is not None:
        return _libri_long_cache
    files = _get_librispeech()
    if not files:
        _libri_long_cache = []
        return _libri_long_cache
    long_files = []
    for f in files:
        try:
            if sf.info(str(f)).duration >= DURATION:
                long_files.append(f)
        except Exception:
            continue
    _libri_long_cache = long_files
    return long_files


def _load_clip(path: Path, n: int, rng: np.random.Generator) -> np.ndarray | None:
    """Load exactly n samples from an audio file.
    For short clips, fades the edges before tiling to avoid click artifacts at
    repetition boundaries."""
    try:
        audio, _ = librosa.load(str(path), sr=SR, mono=True)
    except Exception:
        return None
    if len(audio) == 0:
        return None
    if len(audio) < n:
        # Apply a short fade-in/out at both edges before tiling.
        # Each tile boundary then has a fade-out → fade-in, which sounds
        # like natural amplitude variation rather than a hard click.
        fade = min(int(0.03 * SR), len(audio) // 8)
        if fade > 0:
            ramp = np.linspace(0, 1, fade, dtype=np.float32)
            audio = audio.copy()
            audio[:fade]  *= ramp
            audio[-fade:] *= ramp[::-1]
        reps = (n // len(audio)) + 2
        audio = np.tile(audio, reps).astype(np.float32)
    start = int(rng.integers(0, max(1, len(audio) - n + 1)))
    return audio[start : start + n].astype(np.float32)


# ── Utility generators ─────────────────────────────────────────────────────

def pink_noise(n: int, amp: float = 1.0, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    w = rng.standard_normal(n).astype(np.float32)
    fft = np.fft.rfft(w)
    f = np.fft.rfftfreq(n)
    f[0] = 1.0
    fft[1:] /= np.sqrt(f[1:])
    fft[0] = 0
    out = np.fft.irfft(fft, n=n).astype(np.float32)
    return out / (out.std() + 1e-8) * amp


def band_noise(n: int, lo: float, hi: float, amp: float = 1.0, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    w = rng.standard_normal(n).astype(np.float32)
    nyq = SR / 2.0
    sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    out = sosfilt(sos, w).astype(np.float32)
    return out / (out.std() + 1e-8) * amp


# ── Base audio ─────────────────────────────────────────────────────────────


def _build_speech_single(n: int, rng: np.random.Generator) -> np.ndarray | None:
    """
    Load a single LibriSpeech recording that is already >= DURATION seconds
    and extract an n-sample window from it.  No concatenation, no crossfades,
    no junction artifacts — one continuous natural recording throughout.
    """
    long_files = _get_long_libri_files()
    if not long_files:
        return None
    idx = int(rng.integers(0, len(long_files)))
    try:
        raw, _ = librosa.load(str(long_files[idx]), sr=SR, mono=True)
    except Exception:
        return None
    if len(raw) < n:
        return None
    peak = np.abs(raw).max()
    if peak < 1e-6:
        return None
    raw = (raw / peak * 0.30).astype(np.float32)
    start = int(rng.integers(0, len(raw) - n + 1))
    return raw[start : start + n]


def make_clean(seed: int = 42, duration: float = DURATION) -> np.ndarray:
    """
    A single continuous LibriSpeech recording (>= DURATION s) — one speaker,
    no edits, no joins.  Imperceptible pink noise (0.6% amplitude) is added
    as a stable noise floor for IS divergence to measure against.
    Falls back to synthetic only if LibriSpeech is not available.
    """
    rng = np.random.default_rng(seed)
    n = int(duration * SR)

    speech = _build_speech_single(n, rng)
    if speech is not None:
        # 0.006 ≈ 2% of speech peak — inaudible, gives IS a stable floor to measure
        return (speech + pink_noise(n, 0.006, rng)).astype(np.float32)

    # Fully synthetic fallback
    t = np.linspace(0, n / SR, n, endpoint=False)
    f0 = 150
    carrier = (
        np.sin(2 * np.pi * f0 * t)
        + 0.50 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.25 * np.sin(2 * np.pi * 3 * f0 * t)
    ).astype(np.float32)
    env = ((t % 0.8) < 0.5).astype(np.float32)
    speech_syn = carrier * env * 0.18
    return (pink_noise(n, BG_AMP, rng) + speech_syn).astype(np.float32)


def noise_ref(audio: np.ndarray, dur: float = 0.5) -> np.ndarray:
    """Quietest window of dur seconds — used as noisereduce noise profile."""
    n = int(dur * SR)
    step = max(n // 4, 1)
    best_start, best_rms = 0, float("inf")
    for s in range(0, len(audio) - n, step):
        rms = float(np.sqrt(np.mean(audio[s : s + n] ** 2)))
        if rms < best_rms:
            best_rms, best_start = rms, s
    return audio[best_start : best_start + n]


def denoise_segment(audio: np.ndarray, s: int, e: int, prop: float = 1.0) -> np.ndarray:
    """Apply noisereduce to audio[s:e] and stitch back."""
    ref = noise_ref(audio)
    seg = nr.reduce_noise(
        y=audio[s:e], sr=SR, y_noise=ref, stationary=True, prop_decrease=prop
    ).astype(np.float32)
    out = audio.copy()
    out[s:e] = seg
    return out


# ── ESC-50 noise loader ────────────────────────────────────────────────────

def _esc50_noise(
    noise_type: str, n: int, amp: float, rng: np.random.Generator,
    n_files: int = None,
) -> np.ndarray | None:
    """
    Mix ESC-50 clips for the given noise_type.
    ambient uses n_files=1 (single stationary sound) to keep the background consistent.
    Other types default to 1–3 files for variety.
    Returns None if ESC-50 is not available.
    """
    esc50 = _get_esc50()
    cat_names = ESC50_MAP.get(noise_type, [])
    pool: list[Path] = []
    for cat in cat_names:
        pool.extend(esc50.get(cat, []))
    if not pool:
        return None

    if n_files is None:
        n_files = 1 if noise_type == "ambient" else int(rng.integers(1, min(4, len(pool) + 1)))

    out = np.zeros(n, dtype=np.float32)
    chosen = rng.choice(len(pool), size=min(n_files, len(pool)), replace=False)
    for idx in chosen:
        clip = _load_clip(pool[idx], n, rng)
        if clip is not None:
            out += clip * rng.uniform(0.5, 1.0)

    std = out.std()
    if std < 1e-8:
        return None
    return (out / std) * amp


# ── Noise type functions ───────────────────────────────────────────────────

def crowd_noise(n: int, amp: float, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    result = _esc50_noise("crowd", n, amp, rng)
    return result if result is not None else band_noise(n, 300, 3500, amp, rng)


def human_noise(n: int, amp: float, rng=None) -> np.ndarray:
    """
    Background speech babble: 2–4 random LibriSpeech speakers mixed at low
    volume, simulating a person recording with other people talking nearby.
    Falls back to band-filtered noise if LibriSpeech is unavailable.
    """
    if rng is None:
        rng = np.random.default_rng()
    libri = _get_librispeech()
    if not libri:
        return band_noise(n, 200, 2000, amp, rng)

    n_speakers = int(rng.integers(2, 5))
    order = rng.permutation(len(libri))
    out = np.zeros(n, dtype=np.float32)
    count = 0
    for idx in order:
        if count >= n_speakers:
            break
        try:
            clip, _ = librosa.load(str(libri[idx]), sr=SR, mono=True)
        except Exception:
            continue
        if len(clip) < SR:
            continue
        peak = np.abs(clip).max()
        if peak < 1e-6:
            continue
        clip = (clip / peak).astype(np.float32)
        # Tile with edge fades if too short to fill n samples
        if len(clip) < n:
            fade = min(int(0.03 * SR), len(clip) // 8)
            if fade > 0:
                ramp_e = np.linspace(0, 1, fade, dtype=np.float32)
                clip = clip.copy()
                clip[:fade]  *= ramp_e
                clip[-fade:] *= ramp_e[::-1]
            reps = (n // len(clip)) + 2
            clip = np.tile(clip, reps).astype(np.float32)
        start = int(rng.integers(0, max(1, len(clip) - n + 1)))
        out += clip[start : start + n] * rng.uniform(0.5, 1.0)
        count += 1

    std = out.std()
    if std < 1e-8:
        return band_noise(n, 200, 2000, amp, rng)
    return (out / std) * amp


def white_noise(n: int, amp: float, rng=None) -> np.ndarray:
    """
    Realistic microphone self-noise / preamp hiss.
    Models the hiss a real condenser or dynamic mic produces when recording speech:
      - high-passed above 80 Hz (mics don't capture sub-bass)
      - slight presence boost 2–5 kHz (condenser capsule resonance)
      - pink-noise base (thermal + preamp noise rolls off at high end)
    Much more realistic than flat white noise; this is what you actually
    hear as background hiss in improperly denoised recordings.
    """
    if rng is None:
        rng = np.random.default_rng()
    nyq = SR / 2.0
    # Pink base — thermal noise rolls off at high frequencies
    base = pink_noise(n, amp=1.0, rng=rng)
    # High-pass: mic doesn't capture sub-bass rumble
    sos_hp = butter(2, 80 / nyq, btype="high", output="sos")
    base = sosfilt(sos_hp, base).astype(np.float32)
    # Presence peak 2–5 kHz — condenser capsule / preamp coloration
    presence = band_noise(n, 2000, 5000, amp=0.4, rng=rng)
    out = base + presence
    std = float(out.std())
    if std < 1e-8:
        return rng.standard_normal(n).astype(np.float32) * amp
    return (out / std * amp).astype(np.float32)


def hvac_noise(n: int, amp: float, lo: float = 50, hi: float = 200, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    result = _esc50_noise("hvac", n, amp, rng)
    if result is not None:
        return result
    # Fallback: 50 Hz hum + harmonics + broadband
    t = np.linspace(0, n / SR, n, endpoint=False)
    hum = (
        0.55 * np.sin(2 * np.pi * 50 * t)
        + 0.28 * np.sin(2 * np.pi * 100 * t + 0.12)
        + 0.14 * np.sin(2 * np.pi * 150 * t + 0.25)
        + 0.07 * np.sin(2 * np.pi * 200 * t + 0.08)
    ).astype(np.float32)
    mech = band_noise(n, lo, hi, amp=0.35, rng=rng)
    cycle_rate = rng.uniform(0.3, 0.8)
    cycle = (1.0 + 0.15 * np.sin(2 * np.pi * cycle_rate * t)).astype(np.float32)
    out = (hum + mech) * cycle
    return out / (out.std() + 1e-8) * amp


def outdoor_noise(n: int, amp: float, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    result = _esc50_noise("outdoor", n, amp, rng)
    if result is not None:
        return result
    # Fallback: wind + traffic
    t = np.linspace(0, n / SR, n, endpoint=False)
    wind = band_noise(n, 20, 120, amp=0.5, rng=rng)
    gust_env = np.ones(n, dtype=np.float32)
    for _ in range(int(rng.integers(3, 8))):
        c = rng.integers(0, n)
        w = int(rng.uniform(0.3, 1.5) * SR)
        g = np.exp(-0.5 * ((np.arange(n) - c) / (w / 2)) ** 2)
        gust_env += rng.uniform(0.5, 2.0) * g.astype(np.float32)
    wind = wind * gust_env
    traffic = band_noise(n, 80, 1200, amp=0.6, rng=rng)
    surge = (0.8 + 0.2 * np.abs(np.sin(2 * np.pi * rng.uniform(0.1, 0.3) * t))).astype(np.float32)
    out = (wind + traffic * surge).astype(np.float32)
    return out / (out.std() + 1e-8) * amp


def rain_noise(n: int, amp: float, intensity: int = 500, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    result = _esc50_noise("rain", n, amp, rng)
    if result is not None:
        return result
    # Fallback: Poisson raindrop model
    out = np.zeros(n, dtype=np.float32)
    drop_times = rng.integers(0, n, size=rng.poisson(intensity * n / SR))
    mags = rng.lognormal(0, 0.6, size=len(drop_times)).astype(np.float32)
    decay_len = max(int(0.003 * SR), 1)
    t_dec = np.arange(decay_len) / SR
    for pos, mag in zip(drop_times, mags):
        end = min(int(pos) + decay_len, n)
        ln = end - int(pos)
        out[int(pos) : end] += mag * np.exp(-2500 * t_dec[:ln]) * rng.standard_normal(ln).astype(np.float32)
    nyq = SR / 2.0
    out = sosfilt(butter(2, 800 / nyq, btype="high", output="sos"), out).astype(np.float32)
    out += band_noise(n, 30, 120, amp=0.25, rng=rng)
    out /= out.std() + 1e-8
    return out * amp


# ── Randomised denoising helper ────────────────────────────────────────────

def make_randomised_denoised(
    seed: int = 42, n_windows: int = 3, window_len: float = 1.5, prop: float = 1.0
) -> np.ndarray:
    audio = make_clean(seed)
    rng = np.random.default_rng(seed + 500)
    wl = int(window_len * SR)
    margin = int(1.0 * SR)
    available = list(range(margin, len(audio) - wl - margin))
    starts: list[int] = []
    for _ in range(n_windows):
        candidates = [s for s in available if all(abs(s - p) > wl for p in starts)]
        if not candidates:
            break
        s = int(rng.choice(candidates))
        starts.append(s)
        audio = denoise_segment(audio, s, s + wl, prop)
    return audio


# ── Case registry ──────────────────────────────────────────────────────────

CASES: list[tuple] = []


def reg(path, desc, cat, sub, variant, audio):
    CASES.append((path, desc, cat, sub, variant, audio))


def register_all():
    CASES.clear()

    # ── GENUINE ───────────────────────────────────────────────────────────
    for i, seed in enumerate([42, 99, 7, 13, 256], 1):
        reg(
            f"genuine/genuine_{i:02d}.wav",
            f"Clean genuine (seed {seed})",
            "genuine", "genuine", f"seed_{seed}",
            make_clean(seed),
        )

    # ── NOISE ADDED — CROWD ───────────────────────────────────────────────
    for variant, amp_mult, s, e in [
        ("mild",        2.0, 4.0, 8.0),
        ("moderate",    4.0, 3.0, 8.0),
        ("strong",      7.0, 3.0, 9.0),
        ("short_burst", 5.0, 5.0, 6.5),
    ]:
        clean = make_clean(42)
        si, ei = int(s * SR), int(e * SR)
        added = crowd_noise(ei - si, BG_AMP * amp_mult, np.random.default_rng(1))
        audio = clean.copy(); audio[si:ei] += added
        reg(f"noise_added/crowd/crowd_{variant}.wav",
            f"Crowd noise added — {variant}",
            "noise_added", "crowd", variant, audio.astype(np.float32))

    # ── NOISE ADDED — HUMAN ───────────────────────────────────────────────
    for variant, amp_mult, s, e in [
        ("mild",         1.5, 4.0,  8.0),
        ("moderate",     3.0, 3.0,  8.0),
        ("strong",       5.0, 2.0,  9.0),
        ("short_burst",  4.0, 5.5,  7.0),
        ("long_overlap", 2.5, 1.0, 10.0),
    ]:
        clean = make_clean(42)
        si, ei = int(s * SR), int(e * SR)
        added = human_noise(ei - si, BG_AMP * amp_mult, np.random.default_rng(2))
        audio = clean.copy(); audio[si:ei] += added
        reg(f"noise_added/human/human_{variant}.wav",
            f"Human voice noise added — {variant}",
            "noise_added", "human", variant, audio.astype(np.float32))

    # ── NOISE ADDED — OFFICE ─────────────────────────────────────────────
    for variant, amp_mult, s, e in [
        ("mild",     2.0, 4.0, 8.0),
        ("moderate", 4.0, 3.0, 8.0),
        ("strong",   6.0, 2.0, 9.0),
        ("burst",    8.0, 5.0, 6.0),
    ]:
        clean = make_clean(42)
        si, ei = int(s * SR), int(e * SR)
        added = white_noise(ei - si, BG_AMP * amp_mult, np.random.default_rng(3))
        audio = clean.copy(); audio[si:ei] += added
        reg(f"noise_added/white/white_{variant}.wav",
            f"White noise added — {variant}",
            "noise_added", "white", variant, audio.astype(np.float32))

    # ── NOISE ADDED — HVAC ────────────────────────────────────────────────
    for variant, amp_mult, lo, hi, s, e in [
        ("mild",      2.5,  50, 200, 4.0, 8.0),
        ("moderate",  5.0,  50, 200, 3.0, 8.0),
        ("strong",    8.0,  50, 200, 2.0, 9.0),
        ("high_freq", 4.0, 200, 500, 3.0, 8.0),
    ]:
        clean = make_clean(42)
        si, ei = int(s * SR), int(e * SR)
        added = hvac_noise(ei - si, BG_AMP * amp_mult, lo, hi, np.random.default_rng(4))
        audio = clean.copy(); audio[si:ei] += added
        reg(f"noise_added/hvac/hvac_{variant}.wav",
            f"HVAC noise added — {variant}",
            "noise_added", "hvac", variant, audio.astype(np.float32))

    # ── NOISE ADDED — OUTDOOR ─────────────────────────────────────────────
    for variant, amp_mult, s, e in [
        ("mild",     2.0, 4.0,  8.0),
        ("moderate", 4.0, 3.0,  8.0),
        ("strong",   6.0, 2.0,  9.0),
        ("long",     3.0, 1.0, 10.0),
    ]:
        clean = make_clean(42)
        si, ei = int(s * SR), int(e * SR)
        added = outdoor_noise(ei - si, BG_AMP * amp_mult, np.random.default_rng(5))
        audio = clean.copy(); audio[si:ei] += added
        reg(f"noise_added/outdoor/outdoor_{variant}.wav",
            f"Outdoor ambience added — {variant}",
            "noise_added", "outdoor", variant, audio.astype(np.float32))

    # ── NOISE ADDED — RAIN ────────────────────────────────────────────────
    for variant, amp_mult, intensity, s, e in [
        ("light",  2.0,  100, 4.0, 8.0),
        ("heavy",  5.0, 2000, 3.0, 9.0),
        ("burst",  7.0, 3000, 5.0, 7.0),
        ("long",   3.0,  500, 1.0, 10.0),
    ]:
        clean = make_clean(42)
        si, ei = int(s * SR), int(e * SR)
        added = rain_noise(ei - si, BG_AMP * amp_mult, intensity=intensity,
                           rng=np.random.default_rng(6))
        audio = clean.copy(); audio[si:ei] += added
        reg(f"noise_added/rain/rain_{variant}.wav",
            f"Rain noise added — {variant}",
            "noise_added", "rain", variant, audio.astype(np.float32))

    # ── NOISE REMOVED — START ─────────────────────────────────────────────
    for variant, s, e, prop in [
        ("quarter", 0.0, 3.0, 1.0),
        ("third",   0.0, 4.0, 1.0),
        ("half",    0.0, 6.0, 1.0),
        ("mild",    0.0, 4.0, 0.5),
    ]:
        audio = denoise_segment(make_clean(42), int(s * SR), int(e * SR), prop)
        reg(f"noise_removed/start/start_{variant}.wav",
            f"Denoised at start — {variant}",
            "noise_removed", "start", variant, audio)

    # ── NOISE REMOVED — MID ───────────────────────────────────────────────
    for variant, s, e, prop in [
        ("short",     4.5, 6.5, 1.0),
        ("medium",    3.0, 7.0, 1.0),
        ("long",      2.0, 8.0, 1.0),
        ("mild",      3.0, 7.0, 0.4),
        ("very_mild", 3.0, 7.0, 0.2),
    ]:
        audio = denoise_segment(make_clean(42), int(s * SR), int(e * SR), prop)
        reg(f"noise_removed/mid/mid_{variant}.wav",
            f"Denoised at middle — {variant}",
            "noise_removed", "mid", variant, audio)

    # ── NOISE REMOVED — END ───────────────────────────────────────────────
    for variant, s, e, prop in [
        ("quarter", 9.0, 12.0, 1.0),
        ("third",   8.0, 12.0, 1.0),
        ("half",    6.0, 12.0, 1.0),
        ("mild",    8.0, 12.0, 0.5),
    ]:
        audio = denoise_segment(make_clean(42), int(s * SR), int(e * SR), prop)
        reg(f"noise_removed/end/end_{variant}.wav",
            f"Denoised at end — {variant}",
            "noise_removed", "end", variant, audio)

    # ── NOISE REMOVED — RANDOMISED ────────────────────────────────────────
    for variant, n_win, win_len, prop in [
        ("two_windows",   2, 1.5, 1.0),
        ("three_windows", 3, 1.2, 1.0),
        ("four_windows",  4, 1.0, 1.0),
        ("mild_scatter",  3, 1.2, 0.5),
    ]:
        audio = make_randomised_denoised(seed=42, n_windows=n_win,
                                         window_len=win_len, prop=prop)
        reg(f"noise_removed/randomised/randomised_{variant}.wav",
            f"Randomised denoising — {variant}",
            "noise_removed", "randomised", variant, audio)


# ── Write files ────────────────────────────────────────────────────────────

def build_all():
    # Warm up caches so dataset status is printed once at the top
    _get_librispeech()
    _get_esc50()
    print()

    register_all()

    genuine_count = sum(1 for c in CASES if c[2] == "genuine")
    added_count   = sum(1 for c in CASES if c[2] == "noise_added")
    removed_count = sum(1 for c in CASES if c[2] == "noise_removed")
    print(f"Generating {len(CASES)} test files  "
          f"(genuine={genuine_count}, noise_added={added_count}, noise_removed={removed_count})\n")

    manifest = []
    for rel_path, desc, cat, sub, variant, audio in CASES:
        path = OUT_DIR / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), audio, SR, subtype="PCM_16")
        expected = "genuine" if cat == "genuine" else "manipulated"
        label = f"[{'GENUINE':11s}]" if cat == "genuine" else "[MANIPULATED]"
        print(f"  {label}  {rel_path}")
        manifest.append({
            "file":     rel_path,
            "label":    desc,
            "category": cat,
            "sub_type": sub,
            "variant":  variant,
            "expected": expected,
        })

    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nAll {len(CASES)} files written to: {OUT_DIR}")
    return manifest


if __name__ == "__main__":
    build_all()
