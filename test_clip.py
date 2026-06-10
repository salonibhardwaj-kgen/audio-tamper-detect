"""
CLIP zero-shot classifier: real vs synthetic audio via spectrogram images.

CLIP encodes the spectrogram image and compares it to text descriptions.
Whichever text description has the highest similarity → that's the verdict.
"""

import torch
import open_clip
from PIL import Image
from pathlib import Path

SPEC_DIR = Path("/Users/salonibhardwaj/Desktop/Noise /datasets/spectrograms")

# Load CLIP model (ViT-B/32 — small, runs on CPU, ~350MB)
print("Loading CLIP model...")
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
model.eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")
print("CLIP ready.\n")

# Text prompts — calibrated for spectrogram visual patterns
PROMPTS = [
    "audio spectrogram with uniform consistent background noise throughout all time regions",
    "audio spectrogram where one time region has different background noise than the rest",
]

LABELS = ["real", "synthetic"]


def classify(image_path: Path) -> dict:
    image  = preprocess(Image.open(image_path)).unsqueeze(0)
    tokens = tokenizer(PROMPTS)

    with torch.no_grad():
        image_feat = model.encode_image(image)
        text_feat  = model.encode_text(tokens)

        # Normalise
        image_feat /= image_feat.norm(dim=-1, keepdim=True)
        text_feat  /= text_feat.norm(dim=-1, keepdim=True)

        # Cosine similarity → softmax → probabilities
        similarity = (100.0 * image_feat @ text_feat.T).softmax(dim=-1)
        probs      = similarity[0].tolist()

    verdict    = LABELS[probs.index(max(probs))]
    confidence = round(max(probs), 4)

    return {
        "real":       round(probs[0], 4),
        "synthetic":  round(probs[1], 4),
        "verdict":    verdict,
        "confidence": confidence,
    }


# ── Run on all spectrograms ────────────────────────────────────────────────────
print(f"{'File':<55}  {'Real':>6}  {'Synth':>6}  {'Verdict':<12}  Conf")
print("-" * 95)

genuine_correct   = 0
manip_correct     = 0
genuine_total     = 0
manip_total       = 0

specs = sorted(SPEC_DIR.glob("*.png"))
for spec in specs:
    r = classify(spec)
    is_genuine = spec.name.startswith("genuine")
    expected   = "real" if is_genuine else "synthetic"
    correct    = "✓" if r["verdict"] == expected else "✗"

    if is_genuine:
        genuine_total += 1
        if r["verdict"] == "real": genuine_correct += 1
    else:
        manip_total += 1
        if r["verdict"] == "synthetic": manip_correct += 1

    print(f"  {spec.name:<53}  {r['real']:>6.3f}  {r['synthetic']:>6.3f}"
          f"  {r['verdict']:<12}  {r['confidence']:.3f}  {correct}")

print("-" * 95)
print(f"\n  Genuine clips   : {genuine_correct}/{genuine_total} correctly labelled real")
print(f"  Manipulated     : {manip_correct}/{manip_total} correctly labelled synthetic")
print(f"  Overall accuracy: {(genuine_correct + manip_correct)}/{genuine_total + manip_total}")
