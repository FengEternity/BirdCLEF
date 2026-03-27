"""LB score progress roadmap toward target."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as FancyArrowPatch
import numpy as np
from pathlib import Path

OUT = Path('docs/analyze/figures')

fig, ax = plt.subplots(figsize=(13, 5))

target = 0.93
milestones = [
    ('V? (Initial)', 0.779, '#999999', 'o', 10),
    ('V19 (Inference Opt)', 0.789, '#4C72B0', 'o', 10),
    ('V14 (A1 +SC 30%)', 0.830, '#55A868', 's', 14),
]

future = [
    ('A2 +Noise', 0.845, '#FFB347'),
    ('B1 +10s Train', 0.870, '#FFB347'),
    ('B2 +GeM', 0.885, '#FFB347'),
    ('C1 +ASL', 0.900, '#FFB347'),
    ('D1 5-fold', 0.915, '#FFB347'),
    ('D2 Pseudo-label', 0.935, '#FFB347'),
]

y_main = 1.0
ax.axhline(y=y_main, color='lightgray', linewidth=3, alpha=0.5, zorder=0)

ax.axvline(x=target, color='#C44E52', linewidth=2, linestyle='--', alpha=0.6, zorder=1)
ax.text(target, 2.35, f'Target\n{target}', ha='center', va='bottom', fontsize=11,
        color='#C44E52', fontweight='bold')
ax.plot(target, y_main, '*', color='#C44E52', markersize=20, zorder=5)

for name, score, color, marker, size in milestones:
    ax.plot(score, y_main, marker, color=color, markersize=size, zorder=5, markeredgecolor='white', markeredgewidth=1.5)
    ax.text(score, y_main - 0.55, f'{score:.3f}', ha='center', va='top', fontsize=10, fontweight='bold', color=color)
    ax.text(score, y_main + 0.45, name, ha='center', va='bottom', fontsize=9, color=color,
            rotation=20, style='italic')

for name, score, color in future:
    ax.plot(score, y_main, 'D', color=color, markersize=8, zorder=4, alpha=0.7,
            markeredgecolor='darkorange', markeredgewidth=0.8)
    ax.text(score, y_main - 0.55, f'{score:.3f}', ha='center', va='top', fontsize=8, color='darkorange', alpha=0.8)
    ax.text(score, y_main + 0.35, name, ha='center', va='bottom', fontsize=7.5, color='darkorange',
            rotation=30, alpha=0.8)

completed = milestones[-1][1] - milestones[0][1]
total = target - milestones[0][1]
pct = completed / total * 100

ax.annotate('', xy=(milestones[-1][1], y_main - 1.2), xytext=(milestones[0][1], y_main - 1.2),
            arrowprops=dict(arrowstyle='<->', color='#55A868', lw=2))
ax.text((milestones[0][1] + milestones[-1][1]) / 2, y_main - 1.45,
        f'Completed: +{completed:.3f} ({pct:.0f}%)', ha='center', va='top',
        fontsize=10, color='#55A868', fontweight='bold')

ax.annotate('', xy=(target, y_main - 1.2), xytext=(milestones[-1][1], y_main - 1.2),
            arrowprops=dict(arrowstyle='<->', color='#C44E52', lw=2))
remaining = target - milestones[-1][1]
ax.text((milestones[-1][1] + target) / 2, y_main - 1.45,
        f'Remaining: {remaining:.3f} ({100-pct:.0f}%)', ha='center', va='top',
        fontsize=10, color='#C44E52', fontweight='bold')

ax.fill_betweenx([y_main - 0.15, y_main + 0.15], milestones[0][1], milestones[-1][1],
                 color='#55A868', alpha=0.15, zorder=0)
ax.fill_betweenx([y_main - 0.15, y_main + 0.15], milestones[-1][1], target,
                 color='#C44E52', alpha=0.08, zorder=0)

ax.set_xlim(0.76, 0.955)
ax.set_ylim(-2.0, 3.0)
ax.set_xlabel('LB Score', fontsize=12)
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#4C72B0', markersize=10, label='Completed'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#55A868', markersize=10, label='Current Best'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#FFB347', markersize=8, label='Planned (estimated)'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='#C44E52', markersize=14, label='Target (0.93)'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)

ax.set_title('BirdCLEF 2026: LB Score Roadmap\n0.779 → 0.830 → 0.93 (Target)',
             fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig(OUT / 'v14_progress_roadmap.png', dpi=150)
plt.close()
print(f'Saved: {OUT}/v14_progress_roadmap.png')
