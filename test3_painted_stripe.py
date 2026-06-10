"""
TEST 3 — Synthetic dark stripe (no audio change, just paint the PNG).
Takes a genuine spectrogram PNG, manually darkens the bottom panel
in the 30-60s region, runs CNN.
If CNN says synthetic → it's truly reading the visual dark stripe pattern
If CNN says real     → it's detecting something in the audio, not the image
"""

import numpy as np, torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw
from pathlib import Path

BASE       = Path(os.environ.get("NOISE_BASE_DIR", str(Path(__file__).parent)))
SPEC_DIR   = BASE / "datasets" / "spectrograms"
MODEL_PATH = BASE / "models" / "spectrogram_cnn.pt"
OUT_DIR    = BASE / "results" / "test_generalization"
OUT_DIR.mkdir(exist_ok=True)

# ── Load a genuine spectrogram ────────────────────────────────────────────────
genuine_specs = sorted(SPEC_DIR.glob("genuine_*.png"))
src = genuine_specs[0]
img = Image.open(src).convert("RGB")
W, H = img.size
print(f"Loaded: {src.name}  ({W}×{H} px)")

# ── Locate the bottom panel ───────────────────────────────────────────────────
# generate_spectrograms.py: height_ratios=[3,1] → bottom panel = bottom 25% of height
# Plus matplotlib margins: bottom panel starts at ~78% down
bottom_panel_top    = int(H * 0.78)
bottom_panel_bottom = H

# ── Paint dark stripe in the 30-60s region of bottom panel ───────────────────
# 120s total → 30s starts at 25% of width, ends at 50%
left_margin  = int(W * 0.07)   # approx left axis margin
plot_width   = int(W * 0.86)   # approx plot area width

stripe_left  = left_margin + int(plot_width * 0.25)   # 30s mark
stripe_right = left_margin + int(plot_width * 0.50)   # 60s mark

# Three variants: very dark, medium dark, slightly dark
for label, darkness in [("very_dark", 5), ("medium_dark", 40), ("slight_dark", 90)]:
    edited = img.copy()
    draw   = ImageDraw.Draw(edited)
    draw.rectangle(
        [stripe_left, bottom_panel_top, stripe_right, bottom_panel_bottom],
        fill=(darkness, darkness, darkness)
    )
    out_path = OUT_DIR / f"test3_painted_{label}.png"
    edited.save(str(out_path))
    print(f"  Saved painted stripe ({label}, brightness={darkness}): {out_path.name}")

# Also save unmodified for reference
ref_path = OUT_DIR / "test3_genuine_reference.png"
img.save(str(ref_path))

# ── Load CNN ──────────────────────────────────────────────────────────────────
cnn = models.resnet18()
cnn.fc = nn.Linear(cnn.fc.in_features, 2)
cnn.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
cnn.eval()

val_tf = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def predict(p):
    im = val_tf(Image.open(p).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        out  = cnn(im)
        prob = torch.softmax(out, dim=1)[0]
        pred = out.argmax(1).item()
    return "real" if pred==0 else "synthetic", prob[0].item(), prob[1].item()

print("\n" + "="*60)
print("TEST 3 — SYNTHETIC PAINTED DARK STRIPE")
print("="*60)
results = {}
for label in ["genuine_reference", "painted_very_dark", "painted_medium_dark", "painted_slight_dark"]:
    p = OUT_DIR / f"test3_{label}.png"
    v, r, s = predict(p)
    results[label] = (v, r, s)
    print(f"  {label:<25} → {v:<10} (real={r:.3f}, synth={s:.3f})")

print()
ref_v = results["genuine_reference"][0]
vd_v  = results["painted_very_dark"][0]
md_v  = results["painted_medium_dark"][0]
sd_v  = results["painted_slight_dark"][0]

if ref_v == "real" and vd_v == "synthetic":
    print("  RESULT: ✓ CNN responds to painted stripe")
    print("          → It IS reading the visual dark stripe pattern")
    if md_v == "synthetic":
        print("          → Even medium darkness triggers it")
    if sd_v == "synthetic":
        print("          → Even slight darkness triggers it — very sensitive")
    else:
        print("          → Slight darkness not enough — needs clear dark region")
else:
    print("  RESULT: ✗ CNN ignores painted stripe → detecting something else")
print("="*60)
