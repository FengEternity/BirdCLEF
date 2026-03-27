"""CV-LB gap comparison: V8 vs V14."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path('docs/analyze/figures')

fig, ax = plt.subplots(figsize=(12, 4.5))

y_v8 = 2.2
y_v14 = 0.8
bar_h = 0.55

cv_v8, lb_v8 = 0.9776, 0.789
cv_v14, lb_v14 = 0.9721, 0.830
gap_v8 = cv_v8 - lb_v8
gap_v14 = cv_v14 - lb_v14

ax.barh(y_v8, cv_v8 - 0.7, left=0.7, height=bar_h, color='#4C72B0', alpha=0.3, edgecolor='#4C72B0', linewidth=1)
ax.barh(y_v8, lb_v8 - 0.7, left=0.7, height=bar_h, color='#4C72B0', alpha=0.7, edgecolor='#4C72B0', linewidth=1)

ax.barh(y_v14, cv_v14 - 0.7, left=0.7, height=bar_h, color='#55A868', alpha=0.3, edgecolor='#55A868', linewidth=1)
ax.barh(y_v14, lb_v14 - 0.7, left=0.7, height=bar_h, color='#55A868', alpha=0.7, edgecolor='#55A868', linewidth=1)

bracket_y_v8 = y_v8 + bar_h/2 + 0.05
ax.annotate('', xy=(lb_v8, bracket_y_v8+0.15), xytext=(cv_v8, bracket_y_v8+0.15),
            arrowprops=dict(arrowstyle='<->', color='#C44E52', lw=2))
ax.text((lb_v8+cv_v8)/2, bracket_y_v8+0.25, f'Gap: {gap_v8:.3f}',
        ha='center', va='bottom', fontsize=12, color='#C44E52', fontweight='bold')

bracket_y_v14 = y_v14 + bar_h/2 + 0.05
ax.annotate('', xy=(lb_v14, bracket_y_v14+0.15), xytext=(cv_v14, bracket_y_v14+0.15),
            arrowprops=dict(arrowstyle='<->', color='#55A868', lw=2))
ax.text((lb_v14+cv_v14)/2, bracket_y_v14+0.25, f'Gap: {gap_v14:.3f}',
        ha='center', va='bottom', fontsize=12, color='#55A868', fontweight='bold')

ax.text(0.695, y_v8, 'V8\n(Baseline)', ha='right', va='center', fontsize=11, fontweight='bold', color='#4C72B0')
ax.text(0.695, y_v14, 'V14\n(+SC 30%)', ha='right', va='center', fontsize=11, fontweight='bold', color='#55A868')

for (cv, lb, y_pos, color) in [(cv_v8, lb_v8, y_v8, '#4C72B0'), (cv_v14, lb_v14, y_v14, '#55A868')]:
    ax.plot(cv, y_pos, 'D', color=color, markersize=8, zorder=5)
    ax.text(cv+0.003, y_pos, f'CV {cv:.4f}', ha='left', va='center', fontsize=10, color=color, fontweight='bold')
    ax.plot(lb, y_pos, 's', color=color, markersize=8, zorder=5)
    ax.text(lb-0.003, y_pos, f'LB {lb:.3f}', ha='right', va='center', fontsize=10, color=color, fontweight='bold')

mid_y = (y_v8 + y_v14) / 2
reduction = gap_v8 - gap_v14
reduction_pct = reduction / gap_v8 * 100
ax.annotate('', xy=(0.99, y_v14+bar_h/2+0.02), xytext=(0.99, y_v8-bar_h/2-0.02),
            arrowprops=dict(arrowstyle='->', color='darkred', lw=2.5, connectionstyle='arc3,rad=0'))
ax.text(1.005, mid_y, f'Gap reduced\n{reduction:.3f} ({reduction_pct:.0f}%)',
        ha='left', va='center', fontsize=11, color='darkred', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE0E0', edgecolor='darkred', alpha=0.8))

efficiency = (lb_v14 - lb_v8) / (cv_v8 - cv_v14)
ax.text(0.76, 0.15, f'Efficiency: 1 CV point → {efficiency:.1f} LB points',
        ha='center', va='center', fontsize=10, color='gray', style='italic')

ax.set_xlim(0.69, 1.06)
ax.set_ylim(-0.1, 3.4)
ax.set_xlabel('Score', fontsize=12)
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.set_title('CV-LB Gap Reduction: V8 Baseline → V14 (A1 Soundscape Mixing)',
             fontsize=14, fontweight='bold', pad=15)

diamond = mpatches.Patch(color='gray', alpha=0.3, label='CV AUC range')
square = mpatches.Patch(color='gray', alpha=0.7, label='LB Score range')
ax.legend(handles=[diamond, square], loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig(OUT / 'v14_cv_lb_gap.png', dpi=150)
plt.close()
print(f'Saved: {OUT}/v14_cv_lb_gap.png')
