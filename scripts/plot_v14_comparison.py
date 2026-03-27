"""V8 vs V14 diagnostic comparison visualizations."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path('docs/analyze/figures')
OUT.mkdir(parents=True, exist_ok=True)

v8 = json.load(open('output/diagnostic/diagnostic_results.json'))
v14 = json.load(open('output/diagnostic/diagnostic_results_v14.json'))

plt.rcParams.update({'font.size': 11, 'figure.dpi': 150, 'savefig.bbox': 'tight'})

# ── Figure 1: Per-class SC AUC comparison ──
fig, ax = plt.subplots(figsize=(10, 5))
classes = ['Aves', 'Amphibia', 'Insecta', 'Mammalia', 'Reptilia']
v8_sc = [v8['class_results'][c]['sc_auc'] or 0 for c in classes]
v14_sc = [v14['class_results'][c]['sc_auc'] or 0 for c in classes]
v8_cv = [v8['class_results'][c]['cv_auc'] or 0 for c in classes]
v14_cv = [v14['class_results'][c]['cv_auc'] or 0 for c in classes]

x = np.arange(len(classes))
w = 0.2
bars1 = ax.bar(x - 1.5*w, v8_cv, w, label='V8 CV AUC', color='#4C72B0', alpha=0.7)
bars2 = ax.bar(x - 0.5*w, v8_sc, w, label='V8 SC AUC', color='#DD8452', alpha=0.7)
bars3 = ax.bar(x + 0.5*w, v14_cv, w, label='V14 CV AUC', color='#4C72B0', alpha=1.0, edgecolor='black', linewidth=0.8)
bars4 = ax.bar(x + 1.5*w, v14_sc, w, label='V14 SC AUC', color='#DD8452', alpha=1.0, edgecolor='black', linewidth=0.8)

for i, (s8, s14) in enumerate(zip(v8_sc, v14_sc)):
    ax.annotate(f'+{s14-s8:.2f}', xy=(i+1.5*w, s14), ha='center', va='bottom',
                fontsize=8, color='darkred', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylabel('AUC')
ax.set_ylim(0, 1.12)
ax.set_title('V8 vs V14: Per-class AUC (CV & Soundscape)')
ax.legend(loc='upper left', fontsize=9)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
plt.savefig(OUT / 'v14_class_auc_comparison.png')
plt.close()
print(f'Saved: {OUT}/v14_class_auc_comparison.png')

# ── Figure 2: Domain shift reversal ──
fig, ax = plt.subplots(figsize=(9, 5))
v8_gaps = [v8['class_results'][c].get('gap') or 0 for c in classes]
v14_gaps = [v14['class_results'][c].get('gap') or 0 for c in classes]

x = np.arange(len(classes))
w = 0.35
bars_v8 = ax.bar(x - w/2, v8_gaps, w, label='V8 Domain Shift', color='#C44E52', alpha=0.8)
bars_v14 = ax.bar(x + w/2, v14_gaps, w, label='V14 Domain Shift', color='#55A868', alpha=0.8)

ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylabel('Domain Shift (CV AUC - SC AUC)')
ax.set_title('Domain Shift Reversal: V8 → V14\n(Positive = worse on soundscape, Negative = better on soundscape)')
ax.legend()

for i, (g8, g14) in enumerate(zip(v8_gaps, v14_gaps)):
    ax.annotate(f'{g8:+.2f}', xy=(i-w/2, g8), ha='center',
                va='bottom' if g8 >= 0 else 'top', fontsize=8, color='#C44E52')
    ax.annotate(f'{g14:+.2f}', xy=(i+w/2, g14), ha='center',
                va='bottom' if g14 >= 0 else 'top', fontsize=8, color='#55A868')

plt.savefig(OUT / 'v14_domain_shift_reversal.png')
plt.close()
print(f'Saved: {OUT}/v14_domain_shift_reversal.png')

# ── Figure 3: Calibration comparison ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

thresholds = [0.01, 0.05, 0.10, 0.20, 0.50, 0.90]
v8_pcts = [37.3, 23.5, 16.5, 10.7, 3.7, 0.4]
v14_pcts = [100.0, 99.7, 98.9, 97.2, 89.4, 74.0]

ax = axes[0]
x = np.arange(len(thresholds))
w = 0.35
ax.bar(x - w/2, v8_pcts, w, label='V8', color='#C44E52', alpha=0.8)
ax.bar(x + w/2, v14_pcts, w, label='V14', color='#55A868', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f'>{t}' for t in thresholds])
ax.set_ylabel('% of Positive Samples')
ax.set_xlabel('Prediction Threshold')
ax.set_title('Positive Sample Detection Rate')
ax.legend()

ax = axes[1]
metrics = ['Pos Mean', 'Sep. Ratio', 'Best F1']
v8_vals = [0.067, 19.6, 0.256]
v14_vals = [0.876, 402.5, 0.908]
improvements = [v14_vals[i]/v8_vals[i] for i in range(3)]

colors = ['#4C72B0', '#DD8452', '#55A868']
x_pos = np.arange(len(metrics))
bars = ax.bar(x_pos, improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(metrics)
ax.set_ylabel('V14 / V8 Ratio')
ax.set_title('Calibration Improvement Ratios')
for i, (imp, v8v, v14v) in enumerate(zip(improvements, v8_vals, v14_vals)):
    ax.annotate(f'{imp:.1f}x\n({v8v}→{v14v})', xy=(i, imp), ha='center', va='bottom', fontsize=9)

fig.suptitle('V8 vs V14: Prediction Calibration on Soundscape', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(OUT / 'v14_calibration_comparison.png')
plt.close()
print(f'Saved: {OUT}/v14_calibration_comparison.png')

# ── Figure 4: Zero-shot species transformation ──
zs_v8 = {z['species']: z for z in v8['zero_shot_results']}
zs_v14 = {z['species']: z for z in v14['zero_shot_results']}
common = sorted(set(zs_v8.keys()) & set(zs_v14.keys()),
                key=lambda s: zs_v14[s]['sc_auc'] - zs_v8[s]['sc_auc'], reverse=True)

top_n = min(15, len(common))
species = common[:top_n]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
y = np.arange(top_n)
v8_aucs = [zs_v8[s]['sc_auc'] for s in species]
v14_aucs = [zs_v14[s]['sc_auc'] for s in species]
ax.barh(y, v8_aucs, 0.4, label='V8 SC AUC', color='#C44E52', alpha=0.7, align='edge')
ax.barh(y+0.4, v14_aucs, 0.4, label='V14 SC AUC', color='#55A868', alpha=0.8, align='edge')
ax.set_yticks(y + 0.4)
labels = [f'{s} ({zs_v8[s]["class"][:4]})' for s in species]
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('Soundscape AUC')
ax.set_title('Zero-shot Species: SC AUC')
ax.legend(loc='lower right', fontsize=9)
ax.invert_yaxis()

ax = axes[1]
v8_preds = [zs_v8[s]['pred_at_pos_mean'] for s in species]
v14_preds = [zs_v14[s]['pred_at_pos_mean'] for s in species]
ax.barh(y, v8_preds, 0.4, label='V8 Pred@pos', color='#C44E52', alpha=0.7, align='edge')
ax.barh(y+0.4, v14_preds, 0.4, label='V14 Pred@pos', color='#55A868', alpha=0.8, align='edge')
ax.set_yticks(y + 0.4)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('Mean Prediction at Positive')
ax.set_title('Zero-shot Species: Prediction Confidence')
ax.legend(loc='lower right', fontsize=9)
ax.invert_yaxis()

fig.suptitle('Zero-shot Species: V8 → V14 Transformation', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(OUT / 'v14_zeroshot_transformation.png')
plt.close()
print(f'Saved: {OUT}/v14_zeroshot_transformation.png')

# ── Figure 5: Overall summary dashboard ──
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0.5, 1.05)
ax.set_ylim(-0.5, 7.5)
ax.axis('off')

rows = [
    ('CV macro AUC',      0.9444, 0.9375, ''),
    ('SC macro AUC',      0.6831, 0.9949, '⚠ includes train data'),
    ('LB Score',          0.789,  0.830,  '✓ true generalization'),
    ('Domain Shift',      0.261, -0.057,  'reversed!'),
    ('Pos Pred Mean',     0.067,  0.876,  ''),
    ('Separation',       19.6,  402.5,    ''),
    ('Best F1',           0.256,  0.908,  ''),
    ('Zero-shot Pred',    1e-5,   0.82,   ''),
]

ax.text(0.50, 7.2, 'Metric', fontweight='bold', fontsize=11, ha='left')
ax.text(0.72, 7.2, 'V8', fontweight='bold', fontsize=11, ha='center', color='#C44E52')
ax.text(0.82, 7.2, 'V14', fontweight='bold', fontsize=11, ha='center', color='#55A868')
ax.text(0.92, 7.2, 'Change', fontweight='bold', fontsize=11, ha='center')

for i, (name, v8v, v14v, note) in enumerate(rows):
    y = 6.5 - i
    ax.text(0.50, y, name, fontsize=10, ha='left')
    ax.text(0.72, y, f'{v8v:.4f}' if isinstance(v8v, float) and v8v < 2 else f'{v8v:.1f}',
            fontsize=10, ha='center', color='#C44E52')
    ax.text(0.82, y, f'{v14v:.4f}' if isinstance(v14v, float) and v14v < 2 else f'{v14v:.1f}',
            fontsize=10, ha='center', color='#55A868')
    delta = v14v - v8v
    color = '#55A868' if (delta > 0 and 'Domain' not in name) or (delta < 0 and 'Domain' in name) else '#C44E52'
    sign = '+' if delta > 0 else ''
    ax.text(0.92, y, f'{sign}{delta:.3f}' if abs(delta) < 10 else f'{sign}{delta:.1f}',
            fontsize=10, ha='center', color=color, fontweight='bold')
    if note:
        ax.text(1.02, y, note, fontsize=8, ha='left', color='gray', style='italic')
    ax.axhline(y=y-0.35, xmin=0.0, xmax=1.0, color='lightgray', linewidth=0.5)

ax.set_title('V8 → V14 (A1 Soundscape Mixing): Complete Diagnostic Comparison',
             fontsize=13, fontweight='bold', pad=20)
plt.savefig(OUT / 'v14_summary_dashboard.png')
plt.close()
print(f'Saved: {OUT}/v14_summary_dashboard.png')

print('\nAll 5 figures generated successfully.')
