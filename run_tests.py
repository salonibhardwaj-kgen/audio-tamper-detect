"""
Evaluation harness for the noise stationarity analyzer.

Runs the IS-divergence detector on every file in test_audio/,
compares results to the manifest ground truth, and prints a results table
plus summary statistics. Also sweeps the detection threshold to find the
optimal operating point.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add parent dir so we can import noise_analyzer even when run from elsewhere
sys.path.insert(0, str(Path(__file__).parent))
from noise_analyzer import analyze_file, IS_THRESHOLD, SFM_THRESHOLD, _adaptive_threshold

AUDIO_DIR  = Path(__file__).parent / "test_audio"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_manifest() -> list[dict]:
    manifest_path = AUDIO_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "manifest.json not found. Run generate_test_data.py first."
        )
    with open(manifest_path) as f:
        return json.load(f)


def run_all(manifest: list[dict], threshold: float = IS_THRESHOLD,
            relative_k: float = None,
            sfm_threshold: float = SFM_THRESHOLD) -> list[dict]:
    results = []
    for entry in manifest:
        path = AUDIO_DIR / entry["file"]
        if not path.exists():
            print(f"  MISSING: {entry['file']}")
            continue
        r = analyze_file(str(path), threshold=threshold, relative_k=relative_k,
                         sfm_threshold=sfm_threshold)
        predicted = "manipulated" if r.get("is_manipulated") else "genuine"
        results.append({
            "file":               entry["file"],
            "label":              entry["label"],
            "category":           entry.get("category", ""),
            "sub_type":           entry.get("sub_type", ""),
            "variant":            entry.get("variant", ""),
            "expected":           entry["expected"],
            "predicted":          predicted,
            "correct":            predicted == entry["expected"],
            "max_is":             r.get("max_divergence",    float("nan")),
            "mean_is":            r.get("mean_divergence",   float("nan")),
            "std_is":             r.get("std_divergence",    float("nan")),
            "median_is":          r.get("median_divergence", float("nan")),
            "mad_is":             r.get("mad_divergence",    float("nan")),
            "sfm_max_dev":        r.get("sfm_max_dev",       float("nan")),
            "sfm_triggered":      r.get("sfm_triggered",     False),
            "row_outlier_ratio":  r.get("row_outlier_ratio",  float("nan")),
            "outlier_segment":    r.get("outlier_segment",    ""),
            "adaptive_threshold": r.get("adaptive_threshold", float("nan")),
            "threshold_mode":     r.get("threshold_mode", "absolute"),
            "n_segs":             r.get("n_valid_segments", 0),
            "verdict":            r.get("verdict", "N/A"),
            "error":              r.get("error") or "",
        })
    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_table(results: list[dict], threshold: float):
    W = 130
    print("\n" + "═" * W)
    print(f" NOISE STATIONARITY ANALYSIS — IS threshold = {threshold}  SFM threshold = {SFM_THRESHOLD}")
    print("═" * W)
    print(f"  {'File':<45}  {'Category':<14}  {'Sub-type':<12}  {'Expected':>12}"
          f"  {'Max IS':>8}  {'SFM dev':>8}  {'Trigger':>8}  {'Segs':>5}  {'Result':>12}  {'OK?':>4}")
    print("─" * W)

    correct = wrong = inconclusive = 0
    for r in results:
        if r["error"]:
            tag = "INCONCLUSIVE"; ok = "?"; inconclusive += 1; trigger = "-"
        else:
            ok  = "✓" if r["correct"] else "✗"
            correct += r["correct"]; wrong += not r["correct"]
            tag = r["predicted"].upper()
            if r.get("sfm_triggered") and not (r["max_is"] > threshold):
                trigger = "SFM"
            elif r.get("sfm_triggered"):
                trigger = "BOTH"
            elif r["max_is"] > threshold:
                trigger = "IS"
            else:
                trigger = "-"

        print(
            f"  {r['file'][:45]:<45}  {r['category']:<14}  {r['sub_type']:<12}"
            f"  {r['expected']:>12}  {r['max_is']:>8.4f}  {r.get('sfm_max_dev', float('nan')):>8.4f}"
            f"  {trigger:>8}  {r['n_segs']:>5}  {tag:>12}  {ok:>4}"
        )

    total = correct + wrong + inconclusive
    print("─" * W)
    print(f"  Correct: {correct}/{total}   Wrong: {wrong}/{total}   "
          f"Inconclusive: {inconclusive}/{total}")

    valid = [r for r in results if not r["error"]]
    if valid:
        tp = sum(1 for r in valid if r["expected"] == "manipulated" and r["predicted"] == "manipulated")
        tn = sum(1 for r in valid if r["expected"] == "genuine"     and r["predicted"] == "genuine")
        fp = sum(1 for r in valid if r["expected"] == "genuine"     and r["predicted"] == "manipulated")
        fn = sum(1 for r in valid if r["expected"] == "manipulated" and r["predicted"] == "genuine")
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else float("nan")
        print(f"\n  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
        print(f"  Precision: {prec:.3f}   Recall: {rec:.3f}   F1: {f1:.3f}")

    print("═" * W + "\n")

    # Per-category breakdown
    cats = {}
    for r in valid:
        k = f"{r['category']}/{r['sub_type']}"
        cats.setdefault(k, {"correct": 0, "total": 0})
        cats[k]["total"] += 1
        cats[k]["correct"] += r["correct"]
    print(f"  {'Category / Sub-type':<35}  {'Correct':>8}  {'Accuracy':>9}")
    print("  " + "─" * 55)
    for k, v in sorted(cats.items()):
        acc = v["correct"] / v["total"]
        print(f"  {k:<35}  {v['correct']:>3}/{v['total']:<4}  {acc:>8.1%}")
    print()

    return correct, wrong, inconclusive


# ── Relative-k sweep ─────────────────────────────────────────────────────────

def k_sweep(manifest: list[dict], k_values=None):
    """Sweep k in the adaptive threshold (median + k*MAD) and find best k."""
    if k_values is None:
        k_values = np.arange(0.5, 6.0, 0.1)

    print("Sweeping relative threshold k (median + k×MAD) …")
    best_acc, best_f1, best_k = 0.0, 0.0, 2.0
    accs, f1s = [], []

    for k in k_values:
        results = run_all(manifest, relative_k=float(k))
        valid   = [r for r in results if not r["error"]]
        if not valid:
            accs.append(0.0); f1s.append(0.0); continue

        tp = sum(1 for r in valid if r["expected"] == "manipulated" and r["predicted"] == "manipulated")
        tn = sum(1 for r in valid if r["expected"] == "genuine"     and r["predicted"] == "genuine")
        fp = sum(1 for r in valid if r["expected"] == "genuine"     and r["predicted"] == "manipulated")
        fn = sum(1 for r in valid if r["expected"] == "manipulated" and r["predicted"] == "genuine")

        acc  = (tp + tn) / len(valid)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        accs.append(acc); f1s.append(f1)
        # Prefer highest accuracy; break ties by F1
        if acc > best_acc or (acc == best_acc and f1 > best_f1):
            best_acc, best_f1, best_k = acc, f1, float(k)

    print(f"  Best accuracy {best_acc:.3f}  F1 {best_f1:.3f}  at k = {best_k:.2f}\n")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(k_values, accs, label="Accuracy", lw=2)
    ax.plot(k_values, f1s,  label="F1 score",  lw=2, linestyle="--")
    ax.axvline(best_k, color="red", linestyle=":", label=f"Best k={best_k:.2f}")
    ax.set_xlabel("k  (adaptive threshold = median + k × MAD)")
    ax.set_ylabel("Score")
    ax.set_title("Detection performance vs. relative threshold k")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "k_sweep.png", dpi=150)
    plt.close(fig)
    print(f"  k-sweep plot saved to results/k_sweep.png")
    return best_k, best_acc, best_f1


# ── Threshold sweep ───────────────────────────────────────────────────────────

def threshold_sweep(manifest: list[dict], thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.02, 1.50, 0.01)

    print("Sweeping detection threshold …")
    best_acc, best_thr = 0.0, IS_THRESHOLD
    accs, f1s = [], []

    for thr in thresholds:
        results = run_all(manifest, threshold=float(thr))
        valid   = [r for r in results if not r["error"]]
        if not valid:
            accs.append(0.0); f1s.append(0.0); continue

        tp = sum(1 for r in valid if r["expected"] == "manipulated" and r["predicted"] == "manipulated")
        tn = sum(1 for r in valid if r["expected"] == "genuine"     and r["predicted"] == "genuine")
        fp = sum(1 for r in valid if r["expected"] == "genuine"     and r["predicted"] == "manipulated")
        fn = sum(1 for r in valid if r["expected"] == "manipulated" and r["predicted"] == "genuine")

        acc  = (tp + tn) / len(valid)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        accs.append(acc); f1s.append(f1)
        if acc > best_acc:
            best_acc, best_thr = acc, float(thr)

    print(f"  Best accuracy {best_acc:.3f} at threshold = {best_thr:.3f}\n")

    # Plot accuracy and F1 vs threshold
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(thresholds, accs, label="Accuracy", lw=2)
    ax.plot(thresholds, f1s,  label="F1 score",  lw=2, linestyle="--")
    ax.axvline(IS_THRESHOLD, color="gray", linestyle=":", label=f"Default threshold ({IS_THRESHOLD})")
    ax.axvline(best_thr,     color="red",  linestyle=":", label=f"Best threshold ({best_thr:.3f})")
    ax.set_xlabel("IS divergence threshold")
    ax.set_ylabel("Score")
    ax.set_title("Detection performance vs. IS divergence threshold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "threshold_sweep.png", dpi=150)
    plt.close(fig)
    print(f"  Threshold sweep plot saved to results/threshold_sweep.png")

    return best_thr, best_acc


# ── Divergence distribution plot ──────────────────────────────────────────────

def plot_distributions(results: list[dict]):
    genuine_vals = [r["max_is"] for r in results
                    if r["expected"] == "genuine" and not np.isnan(r["max_is"]) and not r["error"]]
    manip_vals   = [r["max_is"] for r in results
                    if r["expected"] == "manipulated" and not np.isnan(r["max_is"]) and not r["error"]]

    if not genuine_vals and not manip_vals:
        return

    fig, ax = plt.subplots(figsize=(9, 4))

    bins = np.linspace(0, max(max(genuine_vals or [0]), max(manip_vals or [0])) * 1.1, 30)

    if genuine_vals:
        ax.hist(genuine_vals, bins=bins, alpha=0.6, label="Genuine", color="steelblue")
    if manip_vals:
        ax.hist(manip_vals, bins=bins, alpha=0.6, label="Manipulated", color="tomato")

    ax.axvline(IS_THRESHOLD, color="black", linestyle="--",
               label=f"Threshold ({IS_THRESHOLD})", lw=1.5)
    ax.set_xlabel("Max pairwise IS divergence")
    ax.set_ylabel("Count")
    ax.set_title("IS divergence distribution — genuine vs manipulated")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "is_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Distribution plot saved to results/is_distribution.png")


# ── Per-file divergence heatmaps ──────────────────────────────────────────────

def plot_heatmaps(manifest: list[dict], n_examples: int = 6):
    """Plot segment-level IS divergence matrices for a selection of files."""
    from noise_analyzer import analyze_file as af
    selected = manifest[:n_examples]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, entry in zip(axes, selected):
        path = AUDIO_DIR / entry["file"]
        if not path.exists():
            ax.axis("off"); continue
        r = af(str(path))
        if "divergence_matrix" not in r:
            ax.axis("off"); continue

        mat   = r["divergence_matrix"]
        times = [f"{t:.1f}s" for t in r["segment_times"]]
        im    = ax.imshow(mat, cmap="hot_r", vmin=0)
        ax.set_xticks(range(len(times))); ax.set_xticklabels(times, fontsize=7)
        ax.set_yticks(range(len(times))); ax.set_yticklabels(times, fontsize=7)
        verdict = "MANIP" if r["is_manipulated"] else "OK"
        ax.set_title(
            f"{entry['file'][:30]}\nmax IS={r['max_divergence']:.4f}  [{verdict}]",
            fontsize=8
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(selected):]:
        ax.axis("off")

    fig.suptitle("Pairwise IS divergence matrices (per-segment noise floors)", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "divergence_heatmaps.png", dpi=150)
    plt.close(fig)
    print(f"  Heatmap plot saved to results/divergence_heatmaps.png")


# ── Noise floor overlay plot ──────────────────────────────────────────────────

def plot_noise_floors(manifest: list[dict], n_examples: int = 4):
    """Overlay noise floor spectra from each segment for selected files."""
    from noise_analyzer import analyze_file as af
    import numpy as np

    # Pick 2 genuine + 2 manipulated
    genuine = [e for e in manifest if e["expected"] == "genuine"][:2]
    manip   = [e for e in manifest if e["expected"] == "manipulated"][:2]
    selected = genuine + manip

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)

    for ax, entry in zip(axes, selected):
        path = AUDIO_DIR / entry["file"]
        if not path.exists():
            ax.axis("off"); continue
        r = af(str(path))
        if "noise_floors" not in r or not r["noise_floors"]:
            ax.axis("off"); continue

        freqs_n = len(r["noise_floors"][0])
        x_axis  = np.linspace(0, 11025, freqs_n)  # 0..Nyquist

        for idx, (nf, t) in enumerate(zip(r["noise_floors"], r["segment_times"])):
            ax.semilogy(x_axis, nf, alpha=0.7, lw=1.2, label=f"{t:.1f}s")

        verdict = "MANIP" if r["is_manipulated"] else "GENUINE"
        ax.set_title(
            f"{entry['file'][:28]}\n[{verdict}] max IS={r['max_divergence']:.4f}",
            fontsize=8
        )
        ax.set_xlabel("Frequency (Hz)", fontsize=8)
        ax.set_ylabel("Norm. power", fontsize=8)
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(alpha=0.3, which="both")
        ax.set_xlim(0, 4000)

    fig.suptitle("Noise floor spectra per segment — genuine vs manipulated", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "noise_floor_spectra.png", dpi=150)
    plt.close(fig)
    print(f"  Noise floor plot saved to results/noise_floor_spectra.png")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import csv, json
    manifest = load_manifest()
    print(f"Loaded manifest: {len(manifest)} test cases")

    results = run_all(manifest, threshold=IS_THRESHOLD)
    print_table(results, IS_THRESHOLD)

    plot_distributions(results)
    plot_heatmaps(manifest)
    plot_noise_floors(manifest)

    csv_path = RESULTS_DIR / "analysis_results.csv"
    csv_fields = [
        "file", "category", "sub_type", "variant", "label",
        "expected", "predicted", "correct",
        "max_is", "mean_is", "std_is", "sfm_max_dev", "sfm_triggered",
        "n_segs", "error",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\nResults CSV  →  results/analysis_results.csv  ({len(results)} rows)")

    out_path = RESULTS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"Results JSON →  results/results.json")


if __name__ == "__main__":
    main()
