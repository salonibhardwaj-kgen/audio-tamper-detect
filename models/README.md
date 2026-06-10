# Model Checkpoints

Trained ResNet-18 model weights are **not stored in this repository** due to file size (≈ 43 MB each).

## Download

The production model (`spectrogram_cnn_seg_v3.pt`) is attached to the [GitHub Releases](../../releases) page.

```bash
# Download with curl (replace URL with the latest release asset URL)
mkdir -p models
curl -L -o models/spectrogram_cnn_seg_v3.pt \
  https://github.com/YOUR_USERNAME/audio-tamper-detect/releases/download/v1.0.0/spectrogram_cnn_seg_v3.pt
```

## Train from scratch

To reproduce the model from scratch using the training pipeline:

```bash
python retrain_segment_cnn.py
```

This requires the datasets to be present under `datasets/` (see [datasets/README.md](../datasets/README.md)).

## Model versions

| File | Trained on | Val accuracy | Notes |
|---|---|---|---|
| `spectrogram_cnn.pt` | Hindi genuine + noise addition | — | v1, retired |
| `spectrogram_cnn_seg.pt` | + removal segments | — | v1.5, retired |
| `spectrogram_cnn_seg_v2.pt` | + partial noise | — | v2, retired |
| `spectrogram_cnn_seg_v3.pt` | + Assamese noise addition (180 clips) | **99.85%** | **Production** |

## Architecture

- Backbone: ResNet-18 (ImageNet pre-trained)
- Final layer: `Linear(512 → 2)` (genuine / synthetic)
- Input: 224 × 224 RGB mel spectrogram image
- Device: CUDA / MPS / CPU (auto-detected)
