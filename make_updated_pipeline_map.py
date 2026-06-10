"""
Updated pipeline process map — reflects new architecture:
- No fixed length trim/pad
- Sliding window CNN (30s, step 10s)
- Gemini 2.5 Flash VLM for noise addition
- Change point detection for noise removal
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
from pathlib import Path

OUT = Path("/Users/salonibhardwaj/Desktop/Noise /results/UPDATED_PIPELINE.png")
OUT.parent.mkdir(exist_ok=True)

W, H = 28, 50
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor("white")

CX = 14.0   # main centre
GX = 3.0    # GENUINE branch (far left)
LX = 6.5    # NOISE REMOVAL branch
RX = 21.5   # NOISE ADDITION branch

def box(cx, cy, w, h, txt, fs=12, bold=False, dashed=False):
    ls = (0, (6, 3)) if dashed else "solid"
    rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle="round,pad=0.15",
                          linewidth=2.0, edgecolor="black",
                          facecolor="white", linestyle=ls, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy, txt, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal",
            multialignment="center", zorder=4)

def diamond(cx, cy, w, h, txt, fs=12):
    dx, dy = w/2, h/2
    pts = [(cx, cy+dy), (cx+dx, cy), (cx, cy-dy), (cx-dx, cy)]
    ax.add_patch(Polygon(pts, closed=True, facecolor="white",
                         edgecolor="black", linewidth=2.0, zorder=3))
    ax.text(cx, cy, txt, ha="center", va="center", fontsize=fs,
            fontweight="bold", multialignment="center", zorder=4)

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="black",
                                lw=2.0, mutation_scale=22), zorder=2)

def hline(x1, x2, y):
    ax.plot([x1, x2], [y, y], color="black", lw=2.0, zorder=2)

def vline(x, y1, y2):
    ax.plot([x, x], [y1, y2], color="black", lw=2.0, zorder=2)

def lbl(x, y, txt, fs=11):
    ax.text(x, y, txt, ha="center", va="center",
            fontsize=fs, fontstyle="italic", zorder=5)

# ── Title ──────────────────────────────────────────────────────────────────
ax.text(CX, 49.2, "Audio Manipulation Detection — Updated Pipeline",
        ha="center", va="center", fontsize=18, fontweight="bold")

# ══════════════════════════════════════════════════════════════════════════
# STAGE 1 — Upload
# ══════════════════════════════════════════════════════════════════════════
box(CX, 48.0, 11, 0.9, "User uploads  audio.wav  (any duration)",
    fs=14, bold=True)
arrow(CX, 47.55, CX, 46.75)

# ══════════════════════════════════════════════════════════════════════════
# STAGE 2 — Pre-processing
# ══════════════════════════════════════════════════════════════════════════
box(CX, 46.35, 16, 0.85,
    "Pre-processing\n"
    "Resample → 22,050 Hz   ·   Convert to mono   ·   No length forcing",
    fs=11.5)
arrow(CX, 45.92, CX, 45.1)

# ══════════════════════════════════════════════════════════════════════════
# STAGE 3 — Full clip spectrogram (baseline only)
# ══════════════════════════════════════════════════════════════════════════
box(CX, 44.7, 18, 0.85,
    "Generate full-clip 2-panel mel spectrogram\n"
    "Stored as baseline reference for noise floor comparison  —  NOT sent to CNN",
    fs=11.5)
arrow(CX, 44.27, CX, 43.45)

# ══════════════════════════════════════════════════════════════════════════
# STAGE 4a — Slide window → audio chunks
# ══════════════════════════════════════════════════════════════════════════
box(CX, 43.05, 18, 0.75,
    "STEP 1 of 3 — Slide 30s window across full audio  (step = 10s)\n"
    "Extract overlapping 30s audio chunks:  [0–30s]  [10–40s]  [20–50s]  [30–60s]  ...",
    fs=11.5)
arrow(CX, 42.67, CX, 41.9)

# STAGE 4b — Generate spectrogram per chunk
box(CX, 41.5, 18, 0.75,
    "STEP 2 of 3 — Generate 30s mel spectrogram for each chunk\n"
    "2-panel PNG per window  (independent normalisation per chunk)",
    fs=11.5)
arrow(CX, 41.12, CX, 40.35)

# STAGE 4c — CNN per spectrogram
box(CX, 39.95, 18, 0.75,
    "STEP 3 of 3 — Run segment CNN on each 30s spectrogram\n"
    "Output per window:  real  or  synthetic  +  confidence score",
    fs=11.5)
arrow(CX, 39.57, CX, 38.75)

# ══════════════════════════════════════════════════════════════════════════
# STAGE 5 — Decision: any flagged?
# ══════════════════════════════════════════════════════════════════════════
diamond(CX, 38.15, 7.0, 1.15, "Any windows\nflagged as synthetic?", fs=12)

lbl(GX + 2.0, 38.85, "NO  →", fs=11)
lbl(CX + 0.5, 37.45, "YES  ↓", fs=11)

# ── GENUINE branch ─────────────────────────────────────────────────────────
hline(GX, CX, 38.15)
arrow(GX, 38.15, GX, 37.35)

box(GX, 36.95, 4.8, 0.75,
    "GENUINE\nNo manipulation detected",
    fs=12, bold=True)
arrow(GX, 36.57, GX, 35.9)
box(GX, 35.5, 4.8, 0.75,
    "Show spectrogram\nReturn confidence score",
    fs=11)

# ══════════════════════════════════════════════════════════════════════════
# STAGE 5b — Merge flagged windows
# ══════════════════════════════════════════════════════════════════════════
arrow(CX, 37.57, CX, 36.8)

box(CX, 36.4, 16, 0.75,
    "Merge overlapping flagged windows → suspicious time ranges\n"
    "e.g. windows 20–50s + 30–60s flagged  →  merged region: 20–60s",
    fs=11.5)
arrow(CX, 36.02, CX, 35.2)

# ══════════════════════════════════════════════════════════════════════════
# STAGE 6 — Noise floor direction
# ══════════════════════════════════════════════════════════════════════════
diamond(CX, 34.6, 8.0, 1.15,
        "Noise floor direction\n(suspicious region vs full-clip baseline)",
        fs=12)

lbl(LX + 0.8, 35.3, "Floor  DROPPED", fs=11)
lbl(RX - 0.8, 35.3, "Floor  ROSE", fs=11)

# ══════════════════════════════════════════════════════════════════════════
# NOISE REMOVAL  (LX = 6.5)
# ══════════════════════════════════════════════════════════════════════════
hline(LX, CX, 34.6)
arrow(LX, 34.6, LX, 33.75)

box(LX, 33.3, 8.0, 0.85,
    "NOISE REMOVAL DETECTED",
    fs=13, bold=True)
arrow(LX, 32.87, LX, 32.05)

box(LX, 31.6, 8.0, 0.85,
    "Change point detection\non noise floor within suspicious region\n"
    "→ finds exact frame where floor drops & recovers",
    fs=11)
arrow(LX, 31.17, LX, 30.35)

box(LX, 29.9, 8.0, 0.85,
    "Precise start time  +  end time  (±1s)\n"
    "Classify position in clip:\n"
    "start  /  mid  /  end  /  random",
    fs=11)
arrow(LX, 29.47, LX, 28.65)

box(LX, 28.2, 8.0, 0.85,
    "Cannot identify noise type\n"
    "Signal was destroyed —\n"
    "original noise no longer exists",
    fs=11, dashed=True)
arrow(LX, 27.77, LX, 26.95)

box(LX, 26.5, 8.0, 0.85,
    "OUTPUT\n"
    "Time range  ·  Position (start/mid/end/random)  ·  Manipulation confirmed",
    fs=11)

# ══════════════════════════════════════════════════════════════════════════
# NOISE ADDITION  (RX = 21.5)
# ══════════════════════════════════════════════════════════════════════════
hline(CX, RX, 34.6)
arrow(RX, 34.6, RX, 33.75)

box(RX, 33.3, 8.0, 0.85,
    "NOISE ADDITION DETECTED",
    fs=13, bold=True)
arrow(RX, 32.87, RX, 32.05)

box(RX, 31.6, 8.0, 0.85,
    "Send suspicious region spectrogram\nto  Gemini 2.5 Flash  (VLM)",
    fs=11)
arrow(RX, 31.17, RX, 30.35)

box(RX, 29.9, 8.0, 1.05,
    "VLM reads spectrogram:\n"
    "·  What noise type is visible?\n"
    "·  At what % of time axis does noise start / end?\n"
    "·  Confidence level?",
    fs=11)
arrow(RX, 29.37, RX, 28.55)

box(RX, 28.1, 8.0, 0.85,
    "Convert time axis % → exact timestamps\n"
    "e.g. start=33%, end=67% in window 0–30s\n"
    "→ noise located at  10s – 20s",
    fs=11)
arrow(RX, 27.67, RX, 26.85)

box(RX, 26.4, 8.0, 0.85,
    "Noise type identified\n"
    "hvac  ·  crowd  ·  white_noise  ·  rain  ·  outdoor  ·  human",
    fs=11)
arrow(RX, 25.97, RX, 25.15)

box(RX, 24.7, 8.0, 0.85,
    "OUTPUT\n"
    "Noise type  ·  Precise time range  ·  Confidence score",
    fs=11)

# ══════════════════════════════════════════════════════════════════════════
# CONVERGE → FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════
CONV_Y = 23.75

vline(LX,  26.07, CONV_Y)
vline(RX,  24.27, CONV_Y)
hline(LX, RX, CONV_Y)
arrow(CX, CONV_Y, CX, 22.95)

box(CX, 22.55, 14, 0.85,
    "FINAL REPORT  shown to user",
    fs=14, bold=True)
arrow(CX, 22.12, CX, 21.3)

# Three output cards
C1, C2, C3 = 8.0, CX, 20.0
box(C1, 20.75, 6.5, 0.95,
    "Spectrogram image\nHighlighted time ranges",
    fs=11.5)
box(C2, 20.75, 6.5, 0.95,
    "Verdict  ·  Confidence\nManipulation type",
    fs=11.5)
box(C3, 20.75, 6.5, 0.95,
    "Precise time ranges\nstart/mid/end + noise type",
    fs=11.5)

hline(C1, C3, 21.225)
hline(C1, C3, 20.275)

# ══════════════════════════════════════════════════════════════════════════
# KEY CHANGES BOX
# ══════════════════════════════════════════════════════════════════════════
box(CX, 18.7, 24, 2.9,
    "Key changes from previous pipeline\n\n"
    "✓  No fixed 120s length — processes any audio duration\n"
    "✓  Audio split into segments BEFORE CNN classification\n"
    "✓  Sliding window (10s step) enables sub-segment precision  (e.g. 10s–20s within 0–30s)\n"
    "✓  Cross-boundary detection via overlapping windows  (e.g. noise at 30–40s)\n"
    "✓  Gemini 2.5 Flash VLM replaces ratio-based fingerprint for noise type identification\n"
    "✓  Change point detection gives exact removal boundaries  (±1s precision)",
    fs=11.5, dashed=True)

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig.savefig(str(OUT), dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved → {OUT}")
print(f"Size  → {OUT.stat().st_size // 1024} KB")
