"""ANAL-003 诊断分析可视化脚本"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

DATA_PATH = Path(__file__).parent.parent / 'output' / 'diagnostic' / 'diagnostic_results.json'
FIG_DIR = Path(__file__).parent.parent / 'docs' / 'analyze' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_PATH) as f:
    data = json.load(f)

COLORS = {
    'Aves': '#2196F3',
    'Amphibia': '#4CAF50',
    'Insecta': '#FF9800',
    'Mammalia': '#9C27B0',
    'Reptilia': '#795548',
}

# ────────────────────────────────────────────
# Fig 1: Per-class AUC comparison (CV vs SC)
# ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

classes = ['Aves', 'Amphibia', 'Insecta', 'Mammalia', 'Reptilia']
cv_vals = [data['class_results'][c]['cv_auc'] for c in classes]
sc_vals = [data['class_results'][c]['sc_auc'] for c in classes]

x = np.arange(len(classes))
w = 0.35

bars_cv = ax.bar(x - w/2, [v if v else 0 for v in cv_vals], w,
                 label='CV AUC (clean recordings)', color='#1976D2', alpha=0.85, edgecolor='white')
bars_sc = ax.bar(x + w/2, [v if v else 0 for v in sc_vals], w,
                 label='Soundscape AUC (field recordings)', color='#E65100', alpha=0.85, edgecolor='white')

for i, (cv, sc) in enumerate(zip(cv_vals, sc_vals)):
    if cv:
        ax.text(i - w/2, cv + 0.01, f'{cv:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    if sc:
        ax.text(i + w/2, sc + 0.01, f'{sc:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    if cv and sc:
        gap = cv - sc
        mid = (cv + sc) / 2
        ax.annotate(f'Δ={gap:.3f}', xy=(i, mid), fontsize=8, color='#D32F2F',
                    ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFEBEE', edgecolor='#D32F2F', alpha=0.8))

ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=11)
ax.set_ylabel('Macro AUC')
ax.set_ylim(0, 1.15)
ax.set_title('Domain Shift by Taxonomic Class: CV vs Soundscape AUC')
ax.legend(loc='upper right', fontsize=9)
ax.axhline(y=data['sc_macro_auc'], color='#E65100', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(y=data['cv_macro_auc'], color='#1976D2', linestyle='--', alpha=0.5, linewidth=1)

plt.tight_layout()
fig.savefig(FIG_DIR / 'class_auc_comparison.png', bbox_inches='tight')
print(f'Saved: class_auc_comparison.png')
plt.close()

# ────────────────────────────────────────────
# Fig 2: Species-level CV vs SC AUC scatter
# ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))

cv_auc = data['per_species_cv_auc']
sc_auc = data['per_species_sc_auc']

TAX_MAP = {}
try:
    import pandas as pd
    tax_df = pd.read_csv(Path(__file__).parent.parent / 'data' / 'raw' / 'taxonomy.csv')
    TAX_MAP = dict(zip(tax_df['primary_label'].astype(str), tax_df['class_name']))
except Exception:
    pass

species_both = []
for sp in cv_auc:
    cv = cv_auc[sp]
    sc = sc_auc.get(sp)
    if cv is not None and sc is not None:
        cls = TAX_MAP.get(sp, 'Unknown')
        species_both.append((sp, cv, sc, cls))

for cls_name, color in COLORS.items():
    pts = [(sp, cv, sc) for sp, cv, sc, c in species_both if c == cls_name]
    if pts:
        ax.scatter([p[1] for p in pts], [p[2] for p in pts],
                   c=color, s=50, alpha=0.7, label=cls_name, edgecolors='white', linewidth=0.5)

ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)

extreme = [(sp, cv, sc, cls) for sp, cv, sc, cls in species_both if cv - sc > 0.5]
for sp, cv, sc, cls in extreme[:8]:
    ax.annotate(sp, (cv, sc), fontsize=7, alpha=0.8,
                xytext=(5, -5), textcoords='offset points')

ax.set_xlabel('CV AUC (clean recordings)')
ax.set_ylabel('Soundscape AUC (field recordings)')
ax.set_title('Per-species Domain Shift: CV vs Soundscape AUC')
ax.set_xlim(0.4, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.legend(loc='lower right', fontsize=9)

ax.fill_between([0.4, 1.05], [-0.05, -0.05], [0.4, 1.05], alpha=0.05, color='red')
ax.text(0.9, 0.15, 'Domain Shift\nZone', fontsize=9, color='red', alpha=0.5, ha='center')

plt.tight_layout()
fig.savefig(FIG_DIR / 'species_cv_vs_sc_scatter.png', bbox_inches='tight')
print(f'Saved: species_cv_vs_sc_scatter.png')
plt.close()

# ────────────────────────────────────────────
# Fig 3: Worst 20 species (soundscape AUC)
# ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

all_sc = [(sp, v) for sp, v in sc_auc.items() if v is not None]
all_sc.sort(key=lambda x: x[1])
worst20 = all_sc[:20]

species_names = [sp for sp, _ in worst20]
sc_values = [v for _, v in worst20]
cv_values = [cv_auc.get(sp) for sp in species_names]

y = np.arange(len(species_names))
h = 0.35

bars1 = ax.barh(y - h/2, sc_values, h, label='Soundscape AUC', color='#E65100', alpha=0.85, edgecolor='white')
bars2 = ax.barh(y + h/2, [v if v else 0 for v in cv_values], h,
                label='CV AUC', color='#1976D2', alpha=0.6, edgecolor='white')

for i, (sc, cv) in enumerate(zip(sc_values, cv_values)):
    ax.text(sc + 0.01, i - h/2, f'{sc:.3f}', va='center', fontsize=8, color='#BF360C')
    if cv:
        ax.text(max(cv + 0.01, 0.45), i + h/2, f'{cv:.3f}', va='center', fontsize=8, color='#0D47A1')

cls_labels = [TAX_MAP.get(sp, '?') for sp in species_names]
tick_labels = [f'{sp} ({cls[:4]})' for sp, cls in zip(species_names, cls_labels)]
ax.set_yticks(y)
ax.set_yticklabels(tick_labels, fontsize=9)
ax.set_xlabel('AUC')
ax.set_title('Top 20 Worst Species on Soundscape (with CV AUC for comparison)')
ax.legend(loc='lower right', fontsize=9)
ax.set_xlim(0, 1.15)
ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
ax.invert_yaxis()

plt.tight_layout()
fig.savefig(FIG_DIR / 'worst20_species_bar.png', bbox_inches='tight')
print(f'Saved: worst20_species_bar.png')
plt.close()

# ────────────────────────────────────────────
# Fig 4: Prediction calibration (pos vs neg)
# ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cal = data['calibration']
thresholds = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]
pos_fracs = [37.3, 23.5, 16.5, 10.7, 7.9, 3.7, 1.0, 0.4]

ax1 = axes[0]
ax1.bar(['Positive\n(n=6244)', 'Negative\n(n=339608)'],
        [cal['pos_pred_mean'], cal['neg_pred_mean']],
        color=['#E65100', '#1976D2'], alpha=0.85, edgecolor='white')
ax1.set_ylabel('Mean Predicted Probability')
ax1.set_title(f'Prediction Means (Separation: {cal["separation_ratio"]:.1f}x)')
for i, v in enumerate([cal['pos_pred_mean'], cal['neg_pred_mean']]):
    ax1.text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')

ax2 = axes[1]
ax2.bar([f'>{t}' for t in thresholds], pos_fracs, color='#E65100', alpha=0.85, edgecolor='white')
ax2.set_ylabel('% of Positive Samples')
ax2.set_title('Positive Sample Prediction Distribution')
ax2.set_xlabel('Prediction Threshold')
for i, v in enumerate(pos_fracs):
    ax2.text(i, v + 0.5, f'{v}%', ha='center', fontsize=8, fontweight='bold')
ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
ax2.text(0, 52, '50% baseline', fontsize=8, color='gray')

plt.tight_layout()
fig.savefig(FIG_DIR / 'calibration_analysis.png', bbox_inches='tight')
print(f'Saved: calibration_analysis.png')
plt.close()

# ────────────────────────────────────────────
# Fig 5: Zero-shot species analysis
# ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

zs = data['zero_shot_results']
zs_sorted = sorted(zs, key=lambda x: x['sc_auc'] if x['sc_auc'] else 0, reverse=True)

ax1 = axes[0]
names = [r['species'][:12] for r in zs_sorted]
aucs = [r['sc_auc'] if r['sc_auc'] else 0 for r in zs_sorted]
colors = [COLORS.get(r['class'], '#999') for r in zs_sorted]
ax1.barh(range(len(names)), aucs, color=colors, alpha=0.85, edgecolor='white')
ax1.set_yticks(range(len(names)))
ax1.set_yticklabels(names, fontsize=7)
ax1.set_xlabel('Soundscape AUC')
ax1.set_title('Zero-shot Species: Soundscape AUC\n(ranked best → worst)')
ax1.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
ax1.invert_yaxis()

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=n) for n, c in COLORS.items() if n in set(r['class'] for r in zs)]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=8)

ax2 = axes[1]
pos_means = [r['pred_at_pos_mean'] if r['pred_at_pos_mean'] else 0 for r in zs_sorted]
pos_maxes = [r['pred_at_pos_max'] if r['pred_at_pos_max'] else 0 for r in zs_sorted]
y = range(len(names))
ax2.barh(y, pos_maxes, color='#FF9800', alpha=0.6, label='Max prediction', edgecolor='white')
ax2.barh(y, pos_means, color='#E65100', alpha=0.85, label='Mean prediction', edgecolor='white')
ax2.set_yticks(range(len(names)))
ax2.set_yticklabels(names, fontsize=7)
ax2.set_xlabel('Predicted Probability (at positive samples)')
ax2.set_title('Zero-shot Species: Prediction Probabilities\n(all essentially zero)')
ax2.set_xscale('log')
ax2.set_xlim(1e-8, 1e-3)
ax2.legend(loc='lower right', fontsize=8)
ax2.invert_yaxis()

plt.tight_layout()
fig.savefig(FIG_DIR / 'zeroshot_analysis.png', bbox_inches='tight')
print(f'Saved: zeroshot_analysis.png')
plt.close()

# ────────────────────────────────────────────
# Fig 6: Gap decomposition summary
# ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

components = ['Domain Shift\n(CV→SC)', 'Zero-shot\nSpecies', 'Model\nSimplification', 'Single Fold', 'Post-processing']
estimated = [0.26, 0.03, 0.03, 0.01, 0.01]
colors_gap = ['#D32F2F', '#FF5722', '#FF9800', '#FFC107', '#FFEB3B']

cumulative = np.cumsum([0] + estimated[:-1])
for i, (comp, est, cum, col) in enumerate(zip(components, estimated, cumulative, colors_gap)):
    bar = ax.barh(0, est, left=cum, color=col, edgecolor='white', height=0.5)
    if est > 0.02:
        ax.text(cum + est/2, 0, f'{est:.2f}', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white' if i < 2 else 'black')

ax.set_xlim(0, 0.40)
ax.set_ylim(-0.5, 1.5)
ax.set_xlabel('AUC Gap Contribution')
ax.set_title('LB Score Gap Decomposition: CV 0.944 → LB 0.779')
ax.set_yticks([])

legend_patches = [plt.Rectangle((0,0), 1, 1, facecolor=c) for c in colors_gap]
ax.legend(legend_patches, [f'{comp}: ~{est:.2f}' for comp, est in zip(components, estimated)],
          loc='upper right', fontsize=9)

ax.axvline(x=sum(estimated), color='black', linestyle='--', alpha=0.5)
ax.text(sum(estimated) + 0.005, 0.3, f'Total: ~{sum(estimated):.2f}', fontsize=9)

plt.tight_layout()
fig.savefig(FIG_DIR / 'gap_decomposition.png', bbox_inches='tight')
print(f'Saved: gap_decomposition.png')
plt.close()

# ────────────────────────────────────────────
# Fig 7: Domain shift distribution histogram
# ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

shifts = []
for sp, cv, sc, cls in species_both:
    shifts.append(cv - sc)

ax.hist(shifts, bins=25, color='#D32F2F', alpha=0.7, edgecolor='white')
ax.axvline(x=np.mean(shifts), color='#1976D2', linestyle='--', linewidth=2,
           label=f'Mean shift: {np.mean(shifts):.3f}')
ax.axvline(x=np.median(shifts), color='#4CAF50', linestyle='--', linewidth=2,
           label=f'Median shift: {np.median(shifts):.3f}')
ax.set_xlabel('Domain Shift (CV AUC - Soundscape AUC)')
ax.set_ylabel('Number of Species')
ax.set_title('Distribution of Per-species Domain Shift')
ax.legend(fontsize=10)

n_severe = sum(1 for s in shifts if s > 0.5)
n_moderate = sum(1 for s in shifts if 0.2 < s <= 0.5)
n_mild = sum(1 for s in shifts if 0 < s <= 0.2)
n_negative = sum(1 for s in shifts if s <= 0)
ax.text(0.98, 0.95, f'Severe (>0.5): {n_severe}\nModerate (0.2-0.5): {n_moderate}\n'
        f'Mild (0-0.2): {n_mild}\nNegative: {n_negative}',
        transform=ax.transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
fig.savefig(FIG_DIR / 'domain_shift_distribution.png', bbox_inches='tight')
print(f'Saved: domain_shift_distribution.png')
plt.close()

print('\nAll 7 figures generated successfully!')
print(f'Output directory: {FIG_DIR}')
