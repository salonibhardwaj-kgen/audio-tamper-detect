"""
Clean black-and-white process map — PNG only, high-res for sharing.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
from pathlib import Path

OUT = Path("/Users/salonibhardwaj/Desktop/Noise /results/PROCESS_MAP.png")
OUT.parent.mkdir(exist_ok=True)

W, H = 26, 38
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor("white")

CX = 13.0   # main centre
GX = 2.5    # GENUINE branch (far left, non-overlapping)
LX = 6.5    # NOISE REMOVAL branch
RX = 19.5   # NOISE ADDITION branch

def box(cx, cy, w, h, txt, fs=13, bold=False, dashed=False):
    ls = (0, (6, 3)) if dashed else "solid"
    rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle="round,pad=0.15",
                          linewidth=2.0, edgecolor="black",
                          facecolor="white", linestyle=ls, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy, txt, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal",
            multialignment="center", zorder=4)

def diamond(cx, cy, w, h, txt, fs=13):
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

def lbl(x, y, txt, fs=12):
    ax.text(x, y, txt, ha="center", va="center",
            fontsize=fs, fontstyle="italic", zorder=5)

# ── Title ──────────────────────────────────────────────────────────────────────
ax.text(CX, 37.3, "Audio Manipulation Detection — End-to-End Process",
        ha="center", va="center", fontsize=19, fontweight="bold")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN SPINE (centre column CX=13)
# ══════════════════════════════════════════════════════════════════════════════

# 1. Upload
box(CX, 36.3, 10, 0.95, "User uploads  audio.wav", fs=15, bold=True)
arrow(CX, 35.82, CX, 35.0)

# 2. Pre-processing  (narrower so GX branch never overlaps)
box(CX, 34.6, 14, 0.85,
    "Pre-processing\n"
    "Resample → 22,050 Hz   ·   Convert to mono   ·   Trim / pad to 120 s",
    fs=12)
arrow(CX, 34.17, CX, 33.3)

# 3. Spectrogram
box(CX, 32.9, 16, 0.85,
    "Generate 2-panel mel spectrogram\n"
    "Panel 1: full spectrum (0 – 8 kHz)     "
    "Panel 2: noise-floor zoom (0 – 500 Hz, boosted contrast)",
    fs=12)
arrow(CX, 32.47, CX, 31.55)

# 4. Decision: Real or Synthetic
diamond(CX, 30.95, 7.0, 1.2, "Real  or  Synthetic?\n(CNN v2)", fs=13)

lbl(GX + 2.3, 31.65, "→  REAL", fs=12)
lbl(CX + 0.6, 30.25, "SYNTHETIC  ↓", fs=12)

# ══════════════════════════════════════════════════════════════════════════════
# GENUINE BRANCH  (GX=2.5, box width=4.5 → spans x=0.25 to x=4.75, clear of CX boxes)
# ══════════════════════════════════════════════════════════════════════════════
hline(GX, CX, 30.95)
arrow(GX, 30.95, GX, 30.1)

box(GX, 29.65, 4.5, 0.85,
    "GENUINE\nNo manipulation detected",
    fs=13, bold=True)

arrow(GX, 29.22, GX, 28.45)

box(GX, 28.0, 4.5, 0.85,
    "Show spectrogram\nReturn confidence score",
    fs=12)

# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC BRANCH (centre → down)
# ══════════════════════════════════════════════════════════════════════════════
arrow(CX, 30.35, CX, 29.5)

# Box starts at CX-7=6 (just clear of GX box which ends at ~4.75)
box(CX, 29.1, 14, 0.85,
    "Split into 4 × 30s segments\n"
    "Run segment CNN:  "
    "seg0 (0–30s)   seg1 (30–60s)   seg2 (60–90s)   seg3 (90–120s)",
    fs=12)

arrow(CX, 28.67, CX, 27.75)

# 5. Decision: Noise floor direction
diamond(CX, 27.15, 7.5, 1.2, "Noise floor direction\nper flagged segment?", fs=13)

lbl(LX + 0.5, 27.85, "Floor  DROPPED", fs=12)
lbl(RX - 0.5, 27.85, "Floor  ROSE", fs=12)

# ══════════════════════════════════════════════════════════════════════════════
# NOISE REMOVAL  (LX = 6.5)
# ══════════════════════════════════════════════════════════════════════════════
hline(LX, CX, 27.15)
arrow(LX, 27.15, LX, 26.25)

box(LX, 25.8, 7.5, 0.85, "NOISE REMOVAL DETECTED", fs=14, bold=True)
arrow(LX, 25.37, LX, 24.5)

box(LX, 24.05, 7.5, 0.85,
    "Identify affected segment(s)\n"
    "0–30s   ·   30–60s   ·   60–90s   ·   90–120s",
    fs=12)
arrow(LX, 23.62, LX, 22.7)

box(LX, 22.15, 7.5, 1.05,
    "Cannot identify noise type\n"
    "Signal was destroyed —\n"
    "original noise no longer exists",
    fs=12, dashed=True)
arrow(LX, 21.62, LX, 20.75)

box(LX, 20.3, 7.5, 0.85,
    "OUTPUT\n"
    "Segment  ·  Time range  ·  Manipulation confirmed",
    fs=12)

# ══════════════════════════════════════════════════════════════════════════════
# NOISE ADDITION  (RX = 19.5)
# ══════════════════════════════════════════════════════════════════════════════
hline(CX, RX, 27.15)
arrow(RX, 27.15, RX, 26.25)

box(RX, 25.8, 7.5, 0.85, "NOISE ADDITION DETECTED", fs=14, bold=True)
arrow(RX, 25.37, RX, 24.5)

box(RX, 24.05, 7.5, 0.85,
    "Identify affected segment(s)\n"
    "0–30s   ·   30–60s   ·   60–90s   ·   90–120s",
    fs=12)
arrow(RX, 23.62, RX, 22.7)

box(RX, 22.15, 7.5, 1.05,
    "Ratio-based fingerprint\n"
    "Sub-band fractions   ·   SFM\n"
    "Cosine similarity vs reference profiles",
    fs=12)
arrow(RX, 21.62, RX, 20.75)

box(RX, 20.3, 7.5, 0.85,
    "Noise type identified\n"
    "hvac  ·  crowd  ·  white_noise  ·  rain  ·  outdoor  ·  human",
    fs=12)
arrow(RX, 19.87, RX, 19.0)

box(RX, 18.55, 7.5, 0.85,
    "OUTPUT\n"
    "Segment  ·  Noise type  ·  Confidence score",
    fs=12)

# ══════════════════════════════════════════════════════════════════════════════
# CONVERGE → FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════
CONV_Y = 17.65

vline(LX,  19.87, CONV_Y)
vline(RX,  18.12, CONV_Y)
hline(LX, RX, CONV_Y)
arrow(CX, CONV_Y, CX, 16.85)

box(CX, 16.45, 13, 0.85, "FINAL REPORT  shown to user", fs=15, bold=True)
arrow(CX, 16.02, CX, 15.2)

# Three output cards
C1, C2, C3 = 7.5, CX, 18.5
box(C1, 14.7, 6.2, 0.95, "Spectrogram image\nHighlighted segments", fs=12)
box(C2, 14.7, 6.2, 0.95, "Verdict  ·  Confidence\nManipulation type", fs=12)
box(C3, 14.7, 6.2, 0.95, "Segment timeline\nTime ranges affected", fs=12)

hline(C1, C3, 15.175)
hline(C1, C3, 14.225)

# ══════════════════════════════════════════════════════════════════════════════
# KEY ASYMMETRY NOTE
# ══════════════════════════════════════════════════════════════════════════════
box(CX, 12.55, 22, 2.4,
    "Key asymmetry\n\n"
    "Noise Removal :   detect ✓     locate ✓     identify noise type ✗"
    "   (signal destroyed — original noise gone)\n\n"
    "Noise Addition :   detect ✓     locate ✓     identify noise type ✓"
    "   (signal still present)",
    fs=12.5, dashed=True)

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig.savefig(str(OUT), dpi=250, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved → {OUT}")
print(f"Size  → {OUT.stat().st_size // 1024} KB")
