# Project Structure

## Purpose

Automated detection of audio manipulation in Indic-language recordings using noise floor analysis and convolutional neural network classification. Targets two tampering classes: **noise removal** (signal erasure) and **noise addition** (signal contamination).

## Architecture Summary

```
Input WAV → Sliding Window Segmentation → 2-Panel Mel Spectrogram
         → ResNet-18 CNN (genuine/synthetic) → Merge Flagged Windows
         → Noise Floor Direction (dropped/rose) → [Removal] Change Point Detection
                                               → [Addition] Gemini VLM Noise ID
```

## Module Dependency Map

```
run_new_pipeline.py          ← main entry point
   ├── noise_analyzer.py     ← IS divergence stationarity module
   ├── vlm_system_prompt.md  ← Gemini forensics system prompt
   └── models/spectrogram_cnn_seg_v3.pt  ← trained ResNet-18

retrain_segment_cnn.py       ← training entry point
   ├── datasets/spectrograms_genuine_segs/
   ├── datasets/spectrograms_noise_addition_segs/
   ├── datasets/spectrograms_noisereduce_segs/
   ├── datasets/spectrograms_audacity_segs/
   ├── datasets/spectrograms_partial_noise_segs/
   └── datasets/spectrograms_assamese_noise_addition_segs/

test_large_scale.py          → imports run_new_pipeline
test_fleurs_and_export_results.py → imports run_new_pipeline
test_vlm_partial_noise.py    → standalone (own spectrogram + Gemini call)
evaluate_removal_pipeline.py → standalone
```

## Folder-by-Folder Breakdown

### Root — Core Pipeline Scripts

| Script | Role | Status |
|---|---|---|
| `run_new_pipeline.py` | Main inference pipeline (sliding window + CNN + VLM) | Production |
| `noise_analyzer.py` | Itakura-Saito divergence stationarity module | Production |
| `retrain_segment_cnn.py` | Train CNN v3 from spectrogram datasets | Production |
| `vlm_system_prompt.md` | Gemini 2.5 Flash forensics system prompt | Production |
| `generate_spectrograms.py` | Generate training spectrogram PNGs from WAVs | Utility |
| `generate_audacity_removal.py` | Create Audacity-style removal training data | Utility |
| `generate_test_data.py` | Create noise addition / removal test WAVs | Utility |
| `generate_report_docx.py` | Generate .docx forensics report | Utility |
| `evaluate_removal_pipeline.py` | Evaluate IS divergence removal detection | Evaluation |
| `cnn_results_full.py` | Export CNN v3 full results to CSV | Evaluation |

### Root — Validation Scripts

| Script | Role |
|---|---|
| `test_large_scale.py` | 1,310-file accuracy test (all categories) |
| `test_fleurs_and_export_results.py` | OOD phone-recording test (FLEURS) |
| `test_vlm_partial_noise.py` | VLM noise type identification test |
| `test_bengali_generalization.py` | Bengali cross-language test |
| `test_detection_summary.py` | Summary detection test across categories |
| `test_gemini_noise_type.py` | Gemini noise type accuracy test |

### Root — Older Experiment Scripts (Archived)

| Script | Note |
|---|---|
| `train_cnn_classifier.py` | CNN v1 training — superseded by retrain_segment_cnn |
| `train_cnn_segment.py` | CNN segment training — superseded |
| `train_cnn_v2.py` | CNN v2 training — superseded |
| `test_cnn_bigger.py` | CNN v2 evaluation — archived |
| `test_cnn_v2_full.py` | CNN v2 full results — archived |
| `test1_new_speaker.py` | Early speaker generalisation test — archived |
| `test2_different_tool.py` | Tool generalisation test — archived |
| `test3_painted_stripe.py` | Paint-stripe attack test — archived |
| `analyze_noise_addition.py` | Noise addition analysis — archived |
| `compare_vlms.py` | VLM comparison — archived |
| `run_cnn_noise_addition.py` | CNN v1 noise addition test — archived |
| `run_cnn_noise_addition_segmentwise.py` | Segment-wise test — archived |
| `probe_rasa.py` | IS divergence probe on Rasa — archived |
| `probe_noise_classifier.py` | Classifier probe — archived |

### Root — Dataset Download Scripts

| Script | Dataset |
|---|---|
| `run_rasa_genuine.py` | AI4Bharat Rasa Hindi studio recordings |
| `run_rasa_noise_removal.py` | Rasa + noise removal manipulation |
| `run_rasa_noise_addition.py` | Rasa + noise addition manipulation |
| `run_indicvoices_genuine.py` | AI4Bharat IndicVoices Hindi |
| `run_vaani_genuine.py` | ARTPARK-IISc Vaani Hindi |
| `run_coraal_genuine.py` | CORAAL African American English field recordings |

### `models/`

| File | Description |
|---|---|
| `spectrogram_cnn_seg_v3.pt` | **Production model** — ResNet-18, 99.85% accuracy |
| `spectrogram_cnn_seg_v2.pt` | Previous version — superseded |
| `spectrogram_cnn_seg.pt` | v1.5 — superseded |
| `spectrogram_cnn_v2.pt` | CNN v2 backbone — superseded |
| `spectrogram_cnn.pt` | CNN v1 — superseded |

### `datasets/`

Not committed (≈ 9.5 GB). See [datasets/README.md](datasets/README.md).

### `results/`

| File | Description |
|---|---|
| `cnn_v3_large_scale_results.csv` | 1,310-file test results (all categories) |
| `cnn_v3_full_results.csv` | Cross-dataset + OOD test results |
| `cnn_noise_removal_results.csv` | Removal-only validation |
| `cnn_v2_noise_addition_segmentwise.csv` | CNN v2 noise addition results |
| `rasa_genuine.csv` / `rasa_noise_removal.csv` / `rasa_noise_addition.csv` | Per-category Rasa results |
| `UPDATED_PIPELINE.png` | Architecture diagram |

## Entry Points

| Task | Command |
|---|---|
| Analyse a WAV file | `python run_new_pipeline.py path/to/file.wav` |
| Retrain CNN | `python retrain_segment_cnn.py` |
| Large-scale evaluation | `python test_large_scale.py` |
| OOD evaluation | `python test_fleurs_and_export_results.py` |
| VLM noise type test | `python test_vlm_partial_noise.py` |
| Generate report | `python generate_report_docx.py` |

## Model Files Used by Production Pipeline

```
run_new_pipeline.py  →  models/spectrogram_cnn_seg_v3.pt
```

## Key Configuration Constants

Defined in `run_new_pipeline.py`:

| Constant | Value | Meaning |
|---|---|---|
| `SR` | 22050 | Sample rate (Hz) |
| `N_MELS` | 128 | Mel filterbanks |
| `FMAX` | 8000 | Max frequency for mel |
| `N_FFT` | 2048 | FFT window |
| `HOP` | 512 | Hop length |
| `WIN_LEN` | 30 | Sliding window length (s) |
| `WIN_STEP` | 10 | Sliding window step (s) |
| `CNN_THRESH` | 0.5 | CNN synthetic confidence threshold |

Change point detection threshold in `noise_floor_direction()`: ±2.0 dB for direction; −5.0 dB baseline offset for change point scan.
