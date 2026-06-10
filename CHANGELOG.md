# Changelog

All notable changes to this project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2025-06-10

### Added

**CNN v3 production model**
- Retrained ResNet-18 with Assamese noise addition segments (180 clips × 6 noise types)
- Final validation accuracy: 99.85% across 1,310 files (Hindi + Assamese)
- WeightedRandomSampler to handle 1:7 genuine:synthetic class imbalance

**Large-scale validation pipeline**
- `test_large_scale.py` — tests all 1,310 pre-made files, saves CSV
- `test_fleurs_and_export_results.py` — adds OOD FLEURS phone-recording test

**VLM noise type identification**
- `test_vlm_partial_noise.py` — Gemini 2.5 Flash identifies noise type from raw spectrograms
- Updated system prompt: open-ended free-text NOISE_TYPE (not limited to 6 types)
- Added TIME_START and TIME_END percentage fields to structured output
- Raw spectrogram replaces difference spectrogram (diff spectrograms degraded VLM accuracy)

**Cross-language support**
- Assamese: genuine, NR removal, noise addition — all validated
- Bengali: genuine, NR removal, noise addition — all validated

**Report generation**
- `generate_report_docx.py` — professional black-and-white .docx forensics report

**Sliding window inference**
- 30s window, 10s step, arbitrary duration input
- Collects clean windows as baseline audio for VLM reference

**VLM confidence return**
- `gemini_identify()` now returns 4 values: noise_type, confidence, precise_start, precise_end

### Changed

- `run_new_pipeline.py`: BASE path now resolved from `__file__` (no hardcoded absolute path)
- `run_new_pipeline.py`: model upgraded to `spectrogram_cnn_seg_v3.pt`
- VLM system prompt: removed 6-type constraint, added free-text noise identification
- All scripts: hardcoded `/Users/salonibhardwaj/Desktop/Noise` paths replaced with `__file__`-relative resolution

### Fixed

- `gemini_identify()` returned 3 values when caller expected 4 — added confidence return
- NR applied to full clip showed direction=ROSE instead of DROPPED — fixed to apply NR to middle segment only (45–75s)
- Bengali clips too short (31.5s, 36.3s) for sliding window — tiled to 120s

### Removed

- All autoencoder files and scripts (IS divergence stationarity approach is more interpretable and accurate)
- Difference spectrogram input to Gemini (degraded accuracy — reverted to raw spectrograms)

---

## [0.3.0] — 2025-05-28

### Added

- CNN v2 (`spectrogram_cnn_seg_v2.pt`) with partial noise addition segments
- `generate_audacity_removal.py` — Audacity-style removal training data
- Gemini 2.5 Flash VLM integration for noise type identification
- Bengali generalisation tests

---

## [0.2.0] — 2025-05-21

### Added

- CNN v1 (`spectrogram_cnn_seg.pt`) with noisereduce and Audacity removal
- IS divergence stationarity module (`noise_analyzer.py`)
- Sliding window inference in `run_new_pipeline.py`
- Noise floor direction analysis (±2.0 dB threshold)
- Change point detection (baseline − 5.0 dB frame scan)

---

## [0.1.0] — 2025-05-10

### Added

- Initial CNN classifier for noise floor spectrogram images
- Rasa Hindi dataset integration
- Noise addition generation (crowd, rain, HVAC, outdoor, white noise, human)
- Spectrogram generation pipeline (128 mel bins, magma + inferno panels)
