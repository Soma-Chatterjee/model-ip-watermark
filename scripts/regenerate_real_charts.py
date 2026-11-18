import os
import json
import numpy as np
import matplotlib.pyplot as plt

out_dir = os.path.join("research_outputs", "real_empirical_benchmarks")
brain_dir = r"C:\Users\ACER\.gemini\antigravity\brain\9f4ea607-72e3-4326-97a3-39c3f5f97a76"
public_dir = os.path.join("frontend", "public")

json_path = os.path.join(out_dir, "real_benchmark_metrics.json")
with open(json_path, "r") as f:
    benchmark_results = json.load(f)

names = [r["display_name"] for r in benchmark_results]
clean_accs = [r["clean_accuracy"] for r in benchmark_results]
wm_accs = [r["watermarked_accuracy"] for r in benchmark_results]
f1_deltas = [r["f1_delta"] for r in benchmark_results]

# Chart 1: Real Clean vs Watermarked Accuracy Retention (Full Range 0 to 115%)
fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
x = np.arange(len(names))
width = 0.35
rects1 = ax.bar(x - width/2, clean_accs, width, label='Original Clean .pth Baseline', color='#2563eb', edgecolor='white', linewidth=1.2)
rects2 = ax.bar(x + width/2, wm_accs, width, label='DeepMark Watermarked .pth (In-Place)', color='#10b981', edgecolor='white', linewidth=1.2)

ax.set_ylabel('Primary Task Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Real Empirical Accuracy Retention Across In-Place Watermarked Models', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=11, fontweight='semibold')
ax.set_ylim(0, 115)
ax.legend(frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1', fontsize=11, loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Value annotations on top of each bar
for rect in rects1:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%',
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#1e3a8a')

for rect in rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%',
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#065f46')

plt.tight_layout()
c1_path = os.path.join(out_dir, "real_accuracy_retention_chart.png")
fig.savefig(c1_path)
fig.savefig(os.path.join(public_dir, "real_accuracy_retention_chart.png"))
fig.savefig(os.path.join(brain_dir, "real_accuracy_retention_chart.png"))
plt.close(fig)
print(f"[Chart 1 Regenerated with visible Green Bars] {c1_path}")
