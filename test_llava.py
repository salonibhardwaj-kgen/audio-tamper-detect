"""
Test LLaVA on genuine vs manipulated spectrogram images.
Run from terminal: python test_llava.py
"""

import requests, base64, json
from pathlib import Path

SPEC_DIR = Path("/Users/salonibhardwaj/Desktop/Noise /datasets/spectrograms")
OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPT = """This image has TWO panels stacked vertically:
- TOP panel: full mel spectrogram of a 120-second audio recording
- BOTTOM panel: zoomed noise floor (low frequencies only, boosted contrast)

Cyan dashed lines mark 30-second boundaries at 30s, 60s, 90s on both panels.
Brighter color = more energy. The bottom panel reveals subtle noise floor changes.

Focus on the BOTTOM panel. Ask yourself:
- Is the color/brightness uniform across all 4 time regions (0-30s, 30-60s, 60-90s, 90-120s)?
- Is any region noticeably darker (noise removed) or brighter (noise added) than the others?

Answer:
1. Is the noise floor consistent across all regions?
2. Which region looks different (if any)?
3. Is this audio genuine (consistent noise floor) or manipulated (one region differs)?
4. Confidence 0.0-1.0.

Reply in JSON only — verdict must be exactly "genuine" or "manipulated":
{
  "consistent": true,
  "changed_region": "none",
  "observation": "one sentence describing what you see in the bottom panel",
  "verdict": "genuine",
  "confidence": 0.0
}"""


def ask_llava(image_path: Path, label: str) -> dict:
    with open(image_path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode()

    payload = {
        "model": "llava-phi3",
        "prompt": PROMPT,
        "images": [b64],
        "stream": False,
        "format": "json"
    }

    print(f"\n  Sending to LLaVA: {image_path.name}")
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()

    raw = resp.json().get("response", "")
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        import re
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        result = json.loads(m.group()) if m else {"raw": raw}

    print(f"  [{label}]")
    print(f"  Consistent   : {result.get('consistent')}")
    print(f"  Changed region: {result.get('changed_region')}")
    print(f"  Observation  : {result.get('observation')}")
    print(f"  Verdict      : {result.get('verdict')}")
    print(f"  Confidence   : {result.get('confidence')}")
    return result


print("=" * 60)
print("LLaVA Spectrogram Analysis")
print("=" * 60)

# Test 1 — Genuine
genuine = SPEC_DIR / "genuine_rasa_Hindi_Male_001.png"
r1 = ask_llava(genuine, "GENUINE — expect consistent=True, verdict=genuine")

# Test 2 — Noise removed at start (0-30s)
manip_start = SPEC_DIR / "manipulated_rasa_Hindi_Male_001_removal_start.png"
r2 = ask_llava(manip_start, "MANIPULATED start — expect changed_region=0-30s")

# Test 3 — Noise removed at mid (45-75s)
manip_mid = SPEC_DIR / "manipulated_rasa_Hindi_Male_001_removal_mid.png"
r3 = ask_llava(manip_mid, "MANIPULATED mid — expect changed_region=30-60s or 45-75s")

# Test 4 — Noise removed at end (90-120s)
manip_end = SPEC_DIR / "manipulated_rasa_Hindi_Male_001_removal_end.png"
r4 = ask_llava(manip_end, "MANIPULATED end — expect changed_region=90-120s")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
for label, result in [
    ("Genuine",          r1),
    ("Manip start",      r2),
    ("Manip mid",        r3),
    ("Manip end",        r4),
]:
    print(f"  {label:<15}  verdict={result.get('verdict','?'):<12}  "
          f"conf={result.get('confidence','?')}  "
          f"region={result.get('changed_region','?')}")
