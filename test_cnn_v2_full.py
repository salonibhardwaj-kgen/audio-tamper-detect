"""
Full test of CNN v2 on all 399 noise-addition WAVs (segment-wise).
Compares white_noise detection rate before (v1) vs after (v2 with 14 dB SNR).
"""

import torch, torch.nn as nn, csv
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from collections import defaultdict

BASE       = Path(os.environ.get("NOISE_BASE_DIR", str(Path(__file__).parent)))
SEG_DIR    = BASE / "datasets" / "spectrograms_noise_addition_segs"
MODEL_V2   = BASE / "models" / "spectrogram_cnn_v2.pt"
OUT_CSV    = BASE / "results" / "cnn_v2_noise_addition_segmentwise.csv"

CODE_TYPE = {
    "hv": "hvac", "wn": "white_noise", "cr": "crowd",
    "ra": "rain",  "ou": "outdoor",    "hu": "human", "no": "genuine"
}

# ── Load v2 model ─────────────────────────────────────────────────────────────
cnn = models.resnet18()
cnn.fc = nn.Linear(cnn.fc.in_features, 2)
cnn.load_state_dict(torch.load(MODEL_V2, weights_only=True))
cnn.eval()

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict(path):
    img = val_tf(Image.open(path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        out  = cnn(img)
        prob = torch.softmax(out, dim=1)[0]
        pred = out.argmax(1).item()
    return {
        "verdict":        "real" if pred == 0 else "synthetic",
        "conf_real":      round(prob[0].item(), 4),
        "conf_synthetic": round(prob[1].item(), 4),
    }

def parse_seg_types(stem):
    after = stem.split("_multi_", 1)[1]
    return [CODE_TYPE.get(c, c) for c in after.split("_")[:4]]

# ── Run all segment spectrograms ──────────────────────────────────────────────
seg_specs = sorted(SEG_DIR.glob("*_seg?.png"))
print(f"Found {len(seg_specs)} segment spectrograms\n")

type_stats = defaultdict(lambda: {"total": 0, "detected": 0, "conf_sum": 0.0})
rows = []

for spec in seg_specs:
    # stem like: rasa_Hindi_Male_001_multi_cr_no_hv_no_seg2
    stem_full = spec.stem                         # ...._seg2
    seg_idx   = int(stem_full[-1])
    wav_stem  = stem_full[:-5]                    # strip _seg2
    seg_types = parse_seg_types(wav_stem)
    actual    = seg_types[seg_idx]
    is_noisy  = actual != "genuine"

    r = predict(spec)
    correct = (r["verdict"] == "synthetic") if is_noisy else (r["verdict"] == "real")

    type_stats[actual]["total"]    += 1
    type_stats[actual]["detected"] += int(correct)
    type_stats[actual]["conf_sum"] += r["conf_synthetic"]

    rows.append({
        "clip":           wav_stem,
        "segment":        seg_idx,
        "actual_type":    actual,
        "is_noisy":       is_noisy,
        "verdict":        r["verdict"],
        "conf_synthetic": r["conf_synthetic"],
        "conf_real":      r["conf_real"],
        "correct":        correct,
    })

# ── Save CSV ──────────────────────────────────────────────────────────────────
fields = ["clip", "segment", "actual_type", "is_noisy",
          "verdict", "conf_synthetic", "conf_real", "correct"]
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"{'Noise Type':<14}  {'Segments':>8}  {'Correct':>7}  {'Rate':>6}  {'Avg Conf':>9}")
print("-" * 55)
for t, s in sorted(type_stats.items()):
    n   = s["total"]
    c   = s["detected"]
    avg = s["conf_sum"] / n
    print(f"  {t:<12}  {n:>8}  {c:>7}  {c/n*100:>5.1f}%  {avg:>9.3f}")

print(f"\n{'─'*55}")
print("\nv1 vs v2 white_noise comparison:")
wn = type_stats["white_noise"]
print(f"  v1 (MIX_RATIO=0.60, SNR 4.4 dB)  : 100.0%  avg conf 0.861")
print(f"  v2 (MIX_RATIO=0.20, SNR 14 dB)   : {wn['detected']/wn['total']*100:.1f}%   avg conf {wn['conf_sum']/wn['total']:.3f}")
print(f"\nCSV saved → {OUT_CSV.name}")
