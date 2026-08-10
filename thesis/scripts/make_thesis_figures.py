import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

OUT = "/workspace/thesis/figures"
ARCHS = ["DeepLabV3+", "MA-Net", "SegFormer"]
COLORS = {"DeepLabV3+": "#2d6e7e", "MA-Net": "#96660a", "SegFormer": "#a23b30"}

# ---------------------------------------------------------------------------
# Fig 1: Component ablation -- GPU utilisation by policy component
# ---------------------------------------------------------------------------
configs = ["None\n(loose)", "Shard\nonly", "Stage\nonly", "Full PADS\n(adaptive)"]
gpu_util = {
    "DeepLabV3+": [61.0, 84.1, 67.6, 79.2],
    "MA-Net":     [84.6, 86.8, 87.0, 83.7],
    "SegFormer":  [42.1, 50.7, 54.3, 41.0],
}
fig, ax = plt.subplots(figsize=(7.5, 4.2))
x = np.arange(len(configs))
width = 0.26
for i, arch in enumerate(ARCHS):
    ax.bar(x + (i - 1) * width, gpu_util[arch], width, label=arch, color=COLORS[arch])
ax.set_xticks(x)
ax.set_xticklabels(configs)
ax.set_ylabel("Mean GPU utilisation (%)")
ax.set_ylim(0, 100)
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.set_title("Component ablation: GPU utilisation by policy configuration")
fig.savefig(f"{OUT}/ablation-components.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 2: Scratch-capacity sweep -- GPU utilisation vs. capacity
# ---------------------------------------------------------------------------
capacities = [0, 10, 25, 50, 100]
scratch_util = {
    "DeepLabV3+": [79.8, 87.2, 24.2, 81.6, 17.3],
    "MA-Net":     [85.8, 85.1, 85.2, 78.6, 86.9],
    "SegFormer":  [52.0, 53.8, 54.0, 56.0, 51.1],
}
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for arch in ARCHS:
    ax.plot(capacities, scratch_util[arch], marker="o", label=arch, color=COLORS[arch], linewidth=2)
ax.set_xlabel("Scratch capacity (% of corpus)")
ax.set_ylabel("Mean GPU utilisation (%)")
ax.set_xticks(capacities)
ax.set_ylim(0, 100)
ax.legend(frameon=False, fontsize=9)
ax.set_title("Scratch-capacity sweep: GPU utilisation vs. staging budget")
ax.annotate("DeepLabV3+ worst\nat full capacity", xy=(100, 17.3), xytext=(72, 35),
            arrowprops=dict(arrowstyle="->", color="#54615a", lw=1), fontsize=8.5, color="#54615a")
fig.savefig(f"{OUT}/ablation-scratch-capacity.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 3: Threshold sensitivity -- measured ratio vs gamma_d, with policy switch
# ---------------------------------------------------------------------------
gammas = [1.0, 1.25, 1.5, 2.0]
ratios = {
    "DeepLabV3+": [0.64, 1.23, 0.65, 1.17],
    "MA-Net":     [1.03, 0.69, 1.18, 0.68],
    "SegFormer":  [1.23, 1.22, 1.32, 1.20],
}
policies = {
    "DeepLabV3+": ["loose", "loose", "loose", "loose"],
    "MA-Net":     ["shard", "loose", "loose", "loose"],
    "SegFormer":  ["shard", "loose", "loose", "loose"],
}
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
for ax, arch in zip(axes, ARCHS):
    ax.plot(gammas, [gammas[i] for i in range(len(gammas))], "--", color="#8a958e", linewidth=1, label="$\\gamma_d$ (threshold)")
    ax.plot(gammas, ratios[arch], marker="o", color=COLORS[arch], linewidth=2, label="measured ratio")
    for gx, ry, pol in zip(gammas, ratios[arch], policies[arch]):
        if pol == "shard":
            ax.scatter([gx], [ry], s=90, facecolors="none", edgecolors="#96660a", linewidths=2, zorder=5)
    ax.set_title(arch, fontsize=11)
    ax.set_xlabel("$\\gamma_d$")
    ax.set_xticks(gammas)
axes[0].set_ylabel("Measured $T_{data}/T_{gpu}$ ratio")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle("Threshold sensitivity: measured ratio vs. decision threshold\n"
              "(circled points switched policy away from loose)", fontsize=10.5, y=1.06)
fig.savefig(f"{OUT}/ablation-threshold-sensitivity.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 4: RQ4 prefetch-depth sweep -- GPU util vs fixed depth
# ---------------------------------------------------------------------------
depths = [1, 2, 4, 8]
depth_util = {
    "DeepLabV3+": [50.3, 60.9, 65.9, 47.3],
    "MA-Net":     [82.6, 82.4, 76.9, 77.9],
    "SegFormer":  [25.1, 39.3, 48.0, 39.4],
}
fig, ax = plt.subplots(figsize=(7, 4.2))
for arch in ARCHS:
    ax.plot(depths, depth_util[arch], marker="o", label=arch, color=COLORS[arch], linewidth=2)
ax.set_xlabel("Fixed prefetch depth $d$")
ax.set_ylabel("Mean GPU utilisation (%)")
ax.set_xticks(depths)
ax.set_ylim(0, 100)
ax.legend(frameon=False, fontsize=9)
ax.set_title("RQ4: GPU utilisation across fixed prefetch depths")
fig.savefig(f"{OUT}/rq4-prefetch-depth.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 5: Profiler-bias diagnostic -- ratio by post-discard window size
# ---------------------------------------------------------------------------
windows = [8, 16, 32, 64, 143]
window_ratio = {
    "DeepLabV3+": [10.62, 10.55, 10.55, 10.46, 10.33],
    "MA-Net":     [12.18, 12.20, 12.23, 12.22, 11.99],
    "SegFormer":  [20.14, 20.39, 20.57, 20.56, 20.24],
}
uncorrected_8batch = {"DeepLabV3+": 1.28, "MA-Net": None, "SegFormer": 1.15}
fig, ax = plt.subplots(figsize=(7.5, 4.4))
for arch in ARCHS:
    ax.plot(windows[:-1], window_ratio[arch][:-1], marker="o", color=COLORS[arch], linewidth=2, label=f"{arch} (discard 5, then measure)")
    ax.scatter([143], [window_ratio[arch][-1]], marker="*", s=160, color=COLORS[arch], zorder=5)
ax.axhline(1.5, color="#8a958e", linestyle="--", linewidth=1, label="$\\gamma_d$=1.5 (default threshold)")
for arch, val in uncorrected_8batch.items():
    if val is not None:
        ax.scatter([8], [val], marker="x", s=90, color=COLORS[arch], zorder=6)
ax.text(8.5, 1.4, "$\\times$ = uncorrected 8-batch\n(no warm-up discard)", fontsize=8, color="#54615a")
ax.set_xlabel("Post-discard measurement window (batches); $\\star$ = full-epoch (143 batches)")
ax.set_ylabel("Measured $T_{data}/T_{gpu}$ ratio")
ax.set_xscale("log")
ax.set_xticks(windows)
ax.set_xticklabels([str(w) for w in windows])
ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
ax.set_title("Profiler-bias diagnostic: discard-then-measure vs. full-epoch ground truth")
fig.savefig(f"{OUT}/profiler-bias-sweep.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 6: RQ3 end-to-end, 5 methods -- GPU utilisation
# ---------------------------------------------------------------------------
methods = ["3. WebDataset", "4. Full pre-stage\n(ceiling)", "5. PADS\n(adaptive)"]
rq3_util = {
    "DeepLabV3+": [75.4, 84.2, 79.5],
    "MA-Net":     [79.1, 83.3, 84.8],
    "SegFormer":  [40.1, 52.8, 39.5],
}
fig, ax = plt.subplots(figsize=(7.5, 4.2))
x = np.arange(len(methods))
width = 0.26
for i, arch in enumerate(ARCHS):
    ax.bar(x + (i - 1) * width, rq3_util[arch], width, label=arch, color=COLORS[arch])
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylabel("Mean GPU utilisation (%)")
ax.set_ylim(0, 100)
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.set_title("RQ3: GPU utilisation, methods 3–5")
fig.savefig(f"{OUT}/rq3-endtoend.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 7: Reproduction gap closure
# ---------------------------------------------------------------------------
sources = ["Main PADS\npipeline\n(uncorrected)", "Faithful replica,\nearly test\n(batch 8)",
           "Faithful replica—\nDeepLabV3+", "Faithful replica—\nSegFormer",
           "Faithful replica—\nMA-Net", "Reference\n(Casini et al.)"]
values = [36, 45, 72.8, 73.3, 73.5, 74.17]
errs = [4, 0, 0.5, 0.7, 0.65, 0.38]
bar_colors = ["#a23b30", "#96660a", "#2d6e7e", "#a23b30", "#96660a", "#182420"]
fig, ax = plt.subplots(figsize=(8.5, 4.4))
xpos = np.arange(len(sources))
ax.bar(xpos, values, yerr=errs, color=bar_colors, capsize=4, alpha=0.85)
ax.set_xticks(xpos)
ax.set_xticklabels(sources, fontsize=8.5)
ax.set_ylabel("Test IoU (%)")
ax.set_ylim(0, 85)
ax.axhline(74.17, color="#182420", linestyle=":", linewidth=1)
ax.set_title("Reproduction gap: from uncorrected pipeline to closed replica")
fig.savefig(f"{OUT}/reproduction-gap-closure.png")
plt.close(fig)

print("All figures written to", OUT)
