# Audio Tampering Detection for Indic Languages

An end-to-end pipeline for detecting audio manipulation in Indic-language recordings. It classifies whether a given audio file is **genuine** or **synthetically manipulated**, identifies the **type of manipulation** (noise removal or noise addition), localises the **exact time region**, and uses a **visual language model** to identify the added noise type when applicable.

Validated at **99.85% accuracy** across 1,310 independently prepared audio files spanning Hindi and Assamese.

---

## Overview

Audio recordings are increasingly used as evidence in legal, journalistic, and compliance workflows. This pipeline addresses two classes of audio tampering:

| Manipulation type | Description | Detection signal |
|---|---|---|
| **Noise removal** | Background ambient noise suppressed (noisereduce, Audacity) | Localised noise floor *drop* |
| **Noise addition** | Artificial background mixed in (crowd, rain, HVAC, traffic…) | Localised noise floor *rise* |

The core detection insight: a genuine recording has a **stationary noise floor**. Tampering creates a detectable discontinuity in the low-frequency energy level that persists across overlapping 30-second analysis windows.

---

## Features

- **Sliding-window CNN inference** — 30s window, 10s step, works on any audio duration
- **ResNet-18 binary classifier** — trained on mel spectrogram images; 99.85% accuracy on 1,310 files
- **Noise floor direction analysis** — distinguishes removal (floor drops) from addition (floor rises)
- **Change point detection** — sub-second precision on manipulation boundaries
- **VLM noise type identification** — Gemini 2.5 Flash identifies noise type from raw spectrograms (crowd, rain, HVAC, traffic, wind, etc.)
- **Multi-language support** — Hindi, Assamese, Bengali validated; Itakura-Saito divergence module is language-agnostic
- **Cross-dataset generalisation** — trained on studio (Rasa), tested on phone recordings (FLEURS)

---

## Architecture

```
Input WAV (any duration)
        │
        ▼
┌───────────────────┐
│  Audio Loading    │  22050 Hz, mono
│  & Normalisation  │
└─────────┬─────────┘
          │
          ▼
┌───────────────────────────────────────────────┐
│           Sliding Window  (30s / 10s step)    │
│  ┌──────────────────────────────────────────┐ │
│  │  2-Panel Mel Spectrogram                 │ │
│  │  • Top  : full spectrum  (magma)         │ │
│  │  • Bottom: noise floor zoom  (inferno)   │ │
│  └──────────────┬───────────────────────────┘ │
│                 │                             │
│  ┌──────────────▼───────────────────────────┐ │
│  │  ResNet-18 CNN  →  genuine / synthetic   │ │
│  └──────────────┬───────────────────────────┘ │
└─────────────────┼─────────────────────────────┘
                  │ flagged windows
                  ▼
      ┌──────────────────────┐
      │  Merge + Localise    │  → time range (e.g. 45s – 75s)
      └──────────┬───────────┘
                 │
       ┌─────────▼──────────┐
       │  Noise Floor       │  delta < −2 dB → REMOVAL
       │  Direction Check   │  delta > +2 dB → ADDITION
       └──────┬──────┬──────┘
              │      │
      REMOVAL │      │ ADDITION
              ▼      ▼
   Change Point    Gemini 2.5 Flash VLM
   Detection       → noise type + time %
   → exact s–e
```

---

## Folder Structure

```
audio-tamper-detect/
├── run_new_pipeline.py              # Main inference entry point
├── noise_analyzer.py                # IS divergence stationarity module
├── retrain_segment_cnn.py           # Train / retrain CNN classifier
├── vlm_system_prompt.md             # Gemini system prompt (forensics)
│
├── generate_spectrograms.py         # Generate spectrogram training images
├── generate_audacity_removal.py     # Create Audacity removal training data
├── generate_test_data.py            # Create noise addition test data
│
├── test_large_scale.py              # 1,310-file accuracy validation
├── test_fleurs_and_export_results.py# OOD phone-recording test
├── test_vlm_partial_noise.py        # VLM noise type identification test
├── evaluate_removal_pipeline.py     # Removal detection evaluation
├── cnn_results_full.py              # Full result export
├── generate_report_docx.py          # Generate .docx forensics report
│
├── run_rasa_genuine.py              # Test genuine Rasa recordings
├── run_rasa_noise_removal.py        # Test Rasa NR removal
├── run_rasa_noise_addition.py       # Test Rasa noise addition
├── run_coraal_genuine.py            # Test CORAAL field recordings
├── run_indicvoices_genuine.py       # Test IndicVoices recordings
├── run_vaani_genuine.py             # Test Vaani recordings
│
├── models/
│   └── README.md                    # Download instructions for checkpoints
├── datasets/
│   └── README.md                    # Dataset setup instructions
├── results/                         # CSV validation results (committed)
│   ├── cnn_v3_large_scale_results.csv
│   ├── cnn_v3_full_results.csv
│   └── UPDATED_PIPELINE.png
│
├── requirements.txt
├── .env.example
└── PROJECT_STRUCTURE.md
```

---

## Installation

```bash
git clone https://github.com/salonibhardwaj-kgen/audio-tamper-detect.git
cd audio-tamper-detect
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**macOS Apple Silicon** — PyTorch MPS acceleration is used automatically when available.

---

## Environment Setup

Copy the example environment file and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Purpose | Required |
|---|---|---|
| `GEMINI_API_KEY` | Gemini 2.5 Flash VLM for noise type ID | Only for addition detection |
| `HF_TOKEN` | HuggingFace token for streaming Rasa / FLEURS | Only for dataset download |
| `NOISE_BASE_DIR` | Override repo root path | Never (auto-detected) |

---

## Downloading the Model

```bash
# Download the production model (v3)
mkdir -p models
curl -L -o models/spectrogram_cnn_seg_v3.pt \
  https://github.com/salonibhardwaj-kgen/audio-tamper-detect/releases/download/v1.0.0/spectrogram_cnn_seg_v3.pt
```

---

## Running Analysis

```bash
# Analyse a single WAV file
export GEMINI_API_KEY=your_key_here
python run_new_pipeline.py path/to/audio.wav
```

Example output:

```
═════════════════════════════════════════════════════════════
  FILE : interview_recording.wav
═════════════════════════════════════════════════════════════
  Duration : 120.0s
  Baseline noise floor : -52.31 dB
  Windows scanned : 10
  Windows flagged : 3

  ── Region 1: 45s – 75s  (conf=0.997) ──
     Noise floor direction : ROSE
     Manipulation type     : NOISE ADDITION
     Approximate range     : 45s – 75s
     Position in clip      : MID
     Noise type            : crowd murmur
     Confidence            : high
     Precise time range    : 47.3s – 73.8s

  VERDICT : SYNTHETIC  (manipulation detected)
─────────────────────────────────────────────────────────────
```

---

## Training Models

```bash
# Retrain CNN v3 from scratch (requires datasets — see datasets/README.md)
python retrain_segment_cnn.py
```

Training configuration:
- Architecture: ResNet-18, ImageNet pre-trained
- Final layer: Linear(512 → 2)
- Epochs: 20, lr = 1e-4, AdamW
- Class balancing: WeightedRandomSampler (1:7 genuine:synthetic ratio)
- Device: auto (MPS → CUDA → CPU)

---

## Evaluating Models

```bash
# Large-scale validation (1,310 files)
python test_large_scale.py

# OOD phone-recording test (FLEURS)
python test_fleurs_and_export_results.py

# VLM noise type identification (requires GEMINI_API_KEY)
GEMINI_API_KEY=your_key python test_vlm_partial_noise.py
```

---

## Generating Reports

```bash
# Generate .docx forensics report from validation results
python generate_report_docx.py
# Output: results/Audio_Tampering_Detection_Report.docx
```

---

## Example Results

### CNN v3 — Large-Scale Validation (1,310 files)

| Category | Language | Files | Correct | Accuracy |
|---|---|---|---|---|
| Genuine | Hindi + Assamese | 106 | 106 | **100.0%** |
| NR Removal | Hindi | 405 | 404 | **99.8%** |
| Audacity Removal | Hindi | 400 | 399 | **99.8%** |
| Noise Addition | Hindi | 399 | 399 | **100.0%** |
| **OVERALL** | — | **1,310** | **1,308** | **99.85%** |

### Cross-Language Generalisation

| Language | Genuine | NR Removal | Noise Addition |
|---|---|---|---|
| Assamese | ✓ | ✓ | ✓ |
| Bengali | ✓ | ✓ | ✓ |

### Out-of-Distribution (FLEURS phone recordings)

| Language | Genuine | Removal | Addition |
|---|---|---|---|
| Hindi | ✗ (FP) | ✓ | ✓ |
| Tamil | ✗ (FP) | ✓ | ✓ |
| Bengali | ✗ (FP) | ✓ | ✓ |

> **Note:** Genuine phone recordings are false-flagged by CNN v3 (trained on studio audio).
> Fix 1 (CNN v4 with FLEURS genuine in training set) is in progress.

---

## Limitations

- **Phone / field recordings**: CNN v3 generates false positives on consumer phone recordings (training distribution mismatch). CNN v4 with FLEURS genuine training data resolves this.
- **Minimum duration**: At least 30 seconds of audio required for one sliding window.
- **Single manipulation per file**: The pipeline identifies the most prominent manipulation region; stacked tampering events are partially characterised.
- **VLM quota**: Free-tier Gemini API is limited to 20 requests/day. A paid tier is needed for production throughput.
- **Noise addition languages**: Assamese noise addition training data is available; Bengali, Tamil, Telugu, Marathi, and Gujarati noise addition spectrograms have not yet been added.

---

## Future Work

- [ ] Train CNN v4 with FLEURS genuine data to fix OOD phone false positives
- [ ] Add noise addition training spectrograms for Tamil, Telugu, Marathi, Gujarati
- [ ] Build highlighted spectrogram output — annotated PNG for forensic reports
- [ ] Systematic VLM validation across all 6 noise types with paid Gemini tier
- [ ] Evaluate on field recordings from news and government archives

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Citation

If you use this work in research, please cite:

```bibtex
@misc{audio-tamper-detect-2025,
  title   = {Audio Tampering Detection for Indic Languages},
  year    = {2025},
  url     = {https://github.com/salonibhardwaj-kgen/audio-tamper-detect}
}
```
