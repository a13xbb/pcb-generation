"""
Plot per-class defect counts for real and synthetic PCB datasets.
Saves one bar chart per dataset + a combined side-by-side comparison.
"""

from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT   = Path(__file__).parent.parent.parent
OUT    = Path(__file__).parent

CLASS_NAMES = ["mouse_bite", "spur", "missing_hole", "open_circuit", "spurious_copper"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

# ── data loading ─────────────────────────────────────────────────────────────

def count_from_labels(label_dir, split_txt=None):
    if split_txt:
        img_paths   = [Path(p.strip()) for p in Path(split_txt).read_text().splitlines() if p.strip()]
        label_paths = [Path(label_dir) / (p.stem + ".txt") for p in img_paths]
    else:
        label_paths = sorted(Path(label_dir).glob("*.txt"))

    counts = Counter()
    for txt in label_paths:
        if not txt.exists():
            continue
        for line in txt.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                counts[int(parts[0])] += 1
    return [counts[i] for i in range(len(CLASS_NAMES))]


real_train   = count_from_labels(ROOT / "pcb-defect-dataset/train/labels",
                                split_txt=ROOT / "pcb-defect-dataset/train_fair.txt")
real_val     = count_from_labels(ROOT / "pcb-defect-dataset/val/labels")
real_test    = count_from_labels(ROOT / "pcb-defect-dataset/test/labels")

synth_base   = ROOT / "generated_dataset/dataset_full_pipe_cluster"
synth_train  = count_from_labels(synth_base / "labels", split_txt=synth_base / "train.txt")
synth_val    = count_from_labels(synth_base / "labels", split_txt=synth_base / "val.txt")

# ── helpers ───────────────────────────────────────────────────────────────────

def bar_chart(ax, counts, title, total_label=True):
    x = np.arange(len(CLASS_NAMES))
    bars = ax.bar(x, counts, color=COLORS, edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in CLASS_NAMES], fontsize=9)
    ax.set_ylabel("Annotation count", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                f"{count:,}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    if total_label:
        ax.text(0.98, 0.97, f"Total: {sum(counts):,}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="#444")
    return bars


# ── Figure 1: Real dataset (train / val / test stacked) ──────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
fig.suptitle("Real Dataset — Class Distribution (train = fair subset)", fontsize=13, fontweight="bold", y=1.01)

bar_chart(axes[0], real_train, f"Train fair ({sum(real_train):,} ann, 3813 imgs)")
bar_chart(axes[1], real_val,   f"Val    ({sum(real_val):,} annotations)")
bar_chart(axes[2], real_test,  f"Test   ({sum(real_test):,} annotations)")

plt.tight_layout()
plt.savefig(OUT / "real_class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: real_class_distribution.png")


# ── Figure 2: Synthetic dataset (train / val) ─────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)
fig.suptitle("Synthetic Dataset — Class Distribution", fontsize=13, fontweight="bold", y=1.01)

bar_chart(axes[0], synth_train, f"Train  ({sum(synth_train):,} annotations)")
bar_chart(axes[1], synth_val,   f"Val    ({sum(synth_val):,} annotations)")

plt.tight_layout()
plt.savefig(OUT / "synth_class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: synth_class_distribution.png")


# ── Figure 3: Side-by-side train comparison ───────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
fig.suptitle("Train Split Comparison — Real vs Synthetic", fontsize=13, fontweight="bold", y=1.01)

bar_chart(axes[0], real_train,  f"Real Train  ({sum(real_train):,} annotations, 3813 images)")
bar_chart(axes[1], synth_train, f"Synth Train ({sum(synth_train):,} annotations, 1547 images)")

# Add imbalance ratio annotation
for ax, counts in zip(axes, [real_train, synth_train]):
    mean_val = np.mean(counts)
    imbalance = max(counts) / min(counts)
    ax.text(0.98, 0.88, f"Max/min ratio: {imbalance:.2f}×",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color="#666",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", edgecolor="#ccc"))

plt.tight_layout()
plt.savefig(OUT / "comparison_class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: comparison_class_distribution.png")

# ── Console summary ───────────────────────────────────────────────────────────

print("\n── Real dataset ─────────────────────────────────────")
for name, tr, va, te in zip(CLASS_NAMES, real_train, real_val, real_test):
    print(f"  {name:<20}  train={tr:>5}  val={va:>4}  test={te:>4}")
print(f"  {'TOTAL':<20}  train={sum(real_train):>5}  val={sum(real_val):>4}  test={sum(real_test):>4}")
print(f"  Max/min ratio (train): {max(real_train)/min(real_train):.2f}×")

print("\n── Synthetic dataset ────────────────────────────────")
for name, tr, va in zip(CLASS_NAMES, synth_train, synth_val):
    print(f"  {name:<20}  train={tr:>5}  val={va:>4}")
print(f"  {'TOTAL':<20}  train={sum(synth_train):>5}  val={sum(synth_val):>4}")
print(f"  Max/min ratio (train): {max(synth_train)/min(synth_train):.2f}×")
