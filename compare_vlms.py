"""
Fair comparison: llava-phi3 vs Qwen2.5-VL vs ResNet18 CNN
Same 10 spectrograms (5 genuine + 5 manipulated) for all models.
"""

import requests, base64, json, random, torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

SPEC_DIR   = Path("/Users/salonibhardwaj/Desktop/Noise /datasets/spectrograms")
MODEL_PATH = Path("/Users/salonibhardwaj/Desktop/Noise /models/spectrogram_cnn.pt")
OLLAMA_URL = "http://localhost:11434/api/generate"

random.seed(42)

# ── Sample: same 10 spectrograms for both models ───────────────────────────────
genuine_specs = sorted(SPEC_DIR.glob("genuine_*.png"))
manip_specs   = sorted(SPEC_DIR.glob("manipulated_*.png"))
test_specs    = random.sample(genuine_specs, 5) + random.sample(manip_specs, 5)
random.shuffle(test_specs)

PROMPT = """This image shows a mel spectrogram of a 120-second audio recording in two panels:
- TOP panel: full mel spectrogram (frequency vs time)
- BOTTOM panel: zoomed noise floor (low frequencies only, boosted contrast)

Cyan dashed lines mark 30-second boundaries at 30s, 60s, 90s.

Focus on the BOTTOM panel. A real (unmodified) recording has a consistent noise floor color across all 4 time regions (0-30s, 30-60s, 60-90s, 90-120s). A synthetic (manipulated) recording has one region that is noticeably darker (noise removed) or brighter (noise added) compared to the rest.

Reply ONLY in valid JSON — verdict must be exactly "real" or "synthetic":
{
  "verdict": "real",
  "confidence": 0.85,
  "reasoning": "one sentence about what you see in the bottom panel"
}"""


# ── llava-phi3 via Ollama ──────────────────────────────────────────────────────
def ask_llava(image_path: Path) -> dict:
    with open(image_path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode()
    payload = {
        "model":  "llava-phi3",
        "prompt": PROMPT,
        "images": [b64],
        "stream": False,
        "format": "json"
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            import re
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            return json.loads(m.group()) if m else {"verdict": "unknown", "confidence": 0.0, "reasoning": raw}
    except Exception as e:
        return {"verdict": "error", "confidence": 0.0, "reasoning": str(e)}


# ── Qwen2.5-VL via Ollama ─────────────────────────────────────────────────────
def ask_qwen(image_path: Path) -> dict:
    with open(image_path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode()
    payload = {
        "model":  "qwen2.5vl:7b",
        "prompt": PROMPT,
        "images": [b64],
        "stream": False,
        "format": "json"
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            import re
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            return json.loads(m.group()) if m else {"verdict": "unknown", "confidence": 0.0, "reasoning": raw}
    except Exception as e:
        return {"verdict": "error", "confidence": 0.0, "reasoning": str(e)}


# ── ResNet18 CNN ───────────────────────────────────────────────────────────────
cnn = models.resnet18()
cnn.fc = nn.Linear(cnn.fc.in_features, 2)
cnn.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
cnn.eval()

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def ask_cnn(image_path: Path) -> dict:
    img  = val_tf(Image.open(image_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        out  = cnn(img)
        prob = torch.softmax(out, dim=1)[0]
        pred = out.argmax(dim=1).item()
    verdict = "real" if pred == 0 else "synthetic"
    conf    = prob[pred].item()
    return {
        "verdict":    verdict,
        "confidence": round(conf, 4),
        "real":       round(prob[0].item(), 4),
        "synthetic":  round(prob[1].item(), 4),
    }


# ── Run evaluation on same 10 images ──────────────────────────────────────────
print(f"Evaluating {len(test_specs)} spectrograms (5 genuine + 5 manipulated)\n")
print(f"{'File':<50}  {'Expected':<10}  {'llava-φ3':>9}  {'Qwen2.5':>9}  {'CNN':>9}  {'L✓':>3}  {'Q✓':>3}  {'C✓':>3}")
print("-" * 110)

results = []
for spec in test_specs:
    expected = "real" if spec.name.startswith("genuine") else "synthetic"

    l = ask_llava(spec)
    q = ask_qwen(spec)
    c = ask_cnn(spec)

    l_correct = "✓" if l.get("verdict") == expected else "✗"
    q_correct = "✓" if q.get("verdict") == expected else "✗"
    c_correct = "✓" if c.get("verdict") == expected else "✗"

    print(f"  {spec.name:<48}  {expected:<10}  "
          f"{l.get('verdict','?'):>9}  {q.get('verdict','?'):>9}  {c.get('verdict','?'):>9}  "
          f"{l_correct:>3}  {q_correct:>3}  {c_correct:>3}")

    results.append({"file": spec.name, "expected": expected, "llava": l, "qwen": q, "cnn": c})

print("-" * 100)

# ── Per-model detailed scores ──────────────────────────────────────────────────
print("\n── llava-phi3 detailed ──")
for r in results:
    l = r["llava"]
    mark = "✓" if l.get("verdict") == r["expected"] else "✗"
    print(f"  {mark}  {r['file']:<50}  verdict={l.get('verdict','?'):<10}  "
          f"conf={l.get('confidence',0):.2f}  | {str(l.get('reasoning',''))[:70]}")

print("\n── Qwen2.5-VL detailed ──")
for r in results:
    q = r["qwen"]
    mark = "✓" if q.get("verdict") == r["expected"] else "✗"
    print(f"  {mark}  {r['file']:<50}  verdict={q.get('verdict','?'):<10}  "
          f"conf={q.get('confidence',0):.2f}  | {str(q.get('reasoning',''))[:70]}")

print("\n── ResNet18 CNN detailed ──")
for r in results:
    c = r["cnn"]
    mark = "✓" if c.get("verdict") == r["expected"] else "✗"
    print(f"  {mark}  {r['file']:<50}  verdict={c.get('verdict','?'):<10}  "
          f"real={c.get('real',0):.2f}  synth={c.get('synthetic',0):.2f}")

# ── Summary ────────────────────────────────────────────────────────────────────
def score(results, key):
    valid    = [r for r in results if r[key].get("verdict") not in ("error", "unknown")]
    correct  = sum(1 for r in valid if r[key].get("verdict") == r["expected"])
    avg_conf = sum(float(r[key].get("confidence", 0)) for r in valid) / len(valid) if valid else 0
    genuine_correct = sum(1 for r in valid if r["expected"] == "real"     and r[key].get("verdict") == "real")
    genuine_total   = sum(1 for r in valid if r["expected"] == "real")
    manip_correct   = sum(1 for r in valid if r["expected"] == "synthetic" and r[key].get("verdict") == "synthetic")
    manip_total     = sum(1 for r in valid if r["expected"] == "synthetic")
    return correct, avg_conf, genuine_correct, genuine_total, manip_correct, manip_total

l_acc, l_conf, l_gc, l_gt, l_mc, l_mt = score(results, "llava")
q_acc, q_conf, q_gc, q_gt, q_mc, q_mt = score(results, "qwen")
c_acc, c_conf, c_gc, c_gt, c_mc, c_mt = score(results, "cnn")

n = len(results)
print(f"\n{'='*65}")
print(f"  {'Model':<14}  {'Overall':>7}  {'Genuine':>8}  {'Manipulated':>11}  {'Avg Conf':>8}")
print(f"  {'-'*60}")
print(f"  {'llava-phi3':<14}  {l_acc}/{n:>5}  {l_gc}/{l_gt:>6}  {l_mc}/{l_mt:>9}  {l_conf:>8.2f}")
print(f"  {'Qwen2.5-VL':<14}  {q_acc}/{n:>5}  {q_gc}/{q_gt:>6}  {q_mc}/{q_mt:>9}  {q_conf:>8.2f}")
print(f"  {'ResNet18 CNN':<14}  {c_acc}/{n:>5}  {c_gc}/{c_gt:>6}  {c_mc}/{c_mt:>9}  {c_conf:>8.2f}")
print(f"{'='*65}")
