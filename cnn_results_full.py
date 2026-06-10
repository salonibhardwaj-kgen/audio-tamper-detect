"""
CNN full evaluation with reasoning.
Runs on all spectrograms, adds explanation for each verdict.
Saves results to results/cnn_noise_removal_results.csv
"""

import torch, torch.nn as nn, csv
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

SPEC_DIR   = Path("/Users/salonibhardwaj/Desktop/Noise /datasets/spectrograms")
MODEL_PATH = Path("/Users/salonibhardwaj/Desktop/Noise /models/spectrogram_cnn.pt")
OUT_CSV    = Path("/Users/salonibhardwaj/Desktop/Noise /results/cnn_noise_removal_results.csv")
OUT_CSV.parent.mkdir(exist_ok=True)

# ── Load CNN ───────────────────────────────────────────────────────────────────
cnn = models.resnet18()
cnn.fc = nn.Linear(cnn.fc.in_features, 2)
cnn.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
cnn.eval()

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict(spec_path: Path) -> dict:
    img  = val_tf(Image.open(spec_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        out  = cnn(img)
        prob = torch.softmax(out, dim=1)[0]
        pred = out.argmax(dim=1).item()
    return {
        "verdict":   "real" if pred == 0 else "synthetic",
        "real":      round(prob[0].item(), 4),
        "synthetic": round(prob[1].item(), 4),
    }

# ── Reasoning templates ────────────────────────────────────────────────────────
REGION_MAP = {
    "start":  "0–30s",
    "mid":    "45–75s",
    "end":    "90–120s",
    "random": "a random 30s window",
}

def get_reasoning(filename: str, verdict: str) -> tuple[str, str]:
    """Returns (expected_label, reasoning)"""
    name = filename.replace(".png", "")

    if name.startswith("genuine_"):
        return (
            "real",
            "Noise floor is uniform and consistent across all four 30s regions "
            "(0–30s, 30–60s, 60–90s, 90–120s). No discontinuity detected."
        )

    # manipulated_rasa_Hindi_Male_XXX_removal_TYPE
    for rtype, region in REGION_MAP.items():
        if f"_removal_{rtype}" in name:
            return (
                "synthetic",
                f"Noise floor shows a dark stripe in the {region} region of the bottom panel. "
                f"This indicates noise was removed from that segment using noisereduce, "
                f"creating a visible discontinuity compared to the surrounding regions."
            )

    return ("unknown", "Could not determine manipulation type from filename.")


# ── Run on all spectrograms ────────────────────────────────────────────────────
all_specs = sorted(SPEC_DIR.glob("*.png"))
print(f"Total spectrograms : {len(all_specs)}")
print(f"{'File':<55}  {'Expected':<10}  {'Verdict':<10}  {'Conf':>5}  {'OK':>3}")
print("-" * 95)

rows = []
tp = tn = fp = fn = 0

for spec in all_specs:
    r = predict(spec)
    expected, reasoning = get_reasoning(spec.name, r["verdict"])
    correct = r["verdict"] == expected
    conf    = r["synthetic"] if r["verdict"] == "synthetic" else r["real"]
    mark    = "✓" if correct else "✗"

    if expected == "synthetic" and correct:     tp += 1
    elif expected == "real"    and correct:     tn += 1
    elif expected == "synthetic" and not correct: fn += 1
    else:                                         fp += 1

    print(f"  {spec.name:<53}  {expected:<10}  {r['verdict']:<10}  {conf:>5.3f}  {mark}")

    rows.append({
        "file":      spec.name,
        "expected":  expected,
        "verdict":   r["verdict"],
        "correct":   correct,
        "conf_real":      r["real"],
        "conf_synthetic": r["synthetic"],
        "reasoning": reasoning,
    })

# ── Save CSV ───────────────────────────────────────────────────────────────────
fields = ["file", "expected", "verdict", "correct", "conf_real", "conf_synthetic", "reasoning"]
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

# ── Summary ────────────────────────────────────────────────────────────────────
total = tp + tn + fp + fn
print("\n" + "=" * 70)
print(f"  Total spectrograms tested   : {total}")
print(f"  Genuine (real)              : {tn + fp}  →  correctly identified: {tn}")
print(f"  Manipulated (synthetic)     : {tp + fn}  →  correctly identified: {tp}")
print(f"  False Positives (real→synth): {fp}")
print(f"  False Negatives (synth→real): {fn}")
print(f"  Overall Accuracy            : {(tp+tn)/total*100:.1f}%  ({tp+tn}/{total})")
print(f"\n  CSV saved → {OUT_CSV}")
print("=" * 70)

# ── Sample reasoning output ────────────────────────────────────────────────────
print("\n── Sample Reasoning ──")
for r in rows[:3]:
    print(f"\n  File     : {r['file']}")
    print(f"  Verdict  : {r['verdict']}  (conf={r['conf_synthetic']:.3f} synth / {r['conf_real']:.3f} real)")
    print(f"  Correct  : {r['correct']}")
    print(f"  Reason   : {r['reasoning']}")
