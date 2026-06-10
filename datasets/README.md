# Datasets

Audio files and generated spectrograms are **not stored in this repository** (total size ≈ 9.5 GB).

## Primary training data

### Rasa (AI4Bharat)
Studio-quality Indic language TTS recordings used for training and evaluation.

```bash
export HF_TOKEN=hf_your_token_here
python probe_rasa.py         # probe a Hindi clip
python run_rasa_genuine.py   # download and test genuine recordings
```

- License: CC BY 4.0
- Terms: https://huggingface.co/datasets/ai4bharat/Rasa

### FLEURS (Google)
Crowdsourced phone/home recordings — used as out-of-distribution test set.

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('google/fleurs', 'hi_in', split='test', streaming=True)
"
```

- License: CC BY 4.0
- Source: https://huggingface.co/datasets/google/fleurs

### ESC-50
Environmental sound classification dataset — used as noise source library.

```bash
# Clone directly
git clone https://github.com/karolpiczak/ESC-50.git datasets/ESC-50-master
```

- License: CC BY-NC 3.0
- Source: https://github.com/karolpiczak/ESC-50

## Expected folder structure

```
datasets/
├── rasa/                          # Genuine Rasa Hindi recordings (120s WAVs)
├── rasa_assamese/                 # Genuine Rasa Assamese recordings
├── rasa_bengali/                  # Genuine Rasa Bengali (tiled to 120s)
├── rasa_manipulated/
│   ├── noise_removal/             # noisereduce-processed WAVs
│   ├── audacity_removal/          # Audacity-processed WAVs
│   └── noise_addition/            # Noise-mixed WAVs
├── fleurs_genuine/                # FLEURS phone recordings (30 files × 6 languages)
├── noise_sources/                 # Crowd, rain, HVAC, outdoor, white noise, human
│   ├── crowd/crowd_murmur.wav
│   ├── rain/rain_combined.wav
│   ├── hvac/hvac_ventilation.wav
│   ├── outdoor/birds_outdoor.wav
│   ├── white_noise/white_noise.wav
│   └── human/office_scene.wav
├── spectrograms_genuine_segs/     # Generated 30s genuine segment PNGs (label 0)
├── spectrograms_noise_addition_segs/   # Generated synthetic PNGs (label 1)
├── spectrograms_noisereduce_segs/ # Generated NR removal PNGs (label 1)
├── spectrograms_audacity_segs/    # Generated Audacity PNGs (label 1)
├── spectrograms_partial_noise_segs/    # Generated partial noise PNGs (label 1)
└── spectrograms_assamese_noise_addition_segs/  # Assamese noise PNGs (label 1)
```

## Generating spectrogram datasets

Once audio files are in place, generate training spectrograms:

```bash
python retrain_segment_cnn.py  # auto-generates all spectrogram folders and trains
```
