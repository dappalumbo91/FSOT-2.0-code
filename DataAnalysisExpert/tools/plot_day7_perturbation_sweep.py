import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 3:
    print('Usage: plot_day7_perturbation_sweep.py <csv> <out_png>')
    sys.exit(1)

csv = sys.argv[1]
out = sys.argv[2]

df = pd.read_csv(csv)
levels = sorted(df['community_damping'].unique())
fig, axes = plt.subplots(len(levels), 2, figsize=(10, 4*len(levels)))
if len(levels) == 1:
    axes = np.expand_dims(axes, 0)

for i, cd in enumerate(levels):
    sub = df[df['community_damping']==cd]
    pivot = sub.pivot_table(index='sigma', columns='threshold', values='reset_rate')
    im = axes[i,0].imshow(pivot.values, origin='lower', aspect='auto', cmap='magma', vmin=0, vmax=100)
    axes[i,0].set_title(f'reset_rate (%) — community_damping={cd}')
    axes[i,0].set_xlabel('threshold')
    axes[i,0].set_ylabel('sigma')
    axes[i,0].set_xticks(np.arange(pivot.shape[1])); axes[i,0].set_xticklabels(pivot.columns.round(2))
    axes[i,0].set_yticks(np.arange(pivot.shape[0])); axes[i,0].set_yticklabels(pivot.index.round(2))
    fig.colorbar(im, ax=axes[i,0])

    pivot2 = sub.pivot_table(index='sigma', columns='threshold', values='mean_reset_strength')
    im2 = axes[i,1].imshow(pivot2.values, origin='lower', aspect='auto', cmap='viridis')
    axes[i,1].set_title(f'mean_reset_strength — community_damping={cd}')
    axes[i,1].set_xlabel('threshold')
    axes[i,1].set_ylabel('sigma')
    axes[i,1].set_xticks(np.arange(pivot2.shape[1])); axes[i,1].set_xticklabels(pivot2.columns.round(2))
    axes[i,1].set_yticks(np.arange(pivot2.shape[0])); axes[i,1].set_yticklabels(pivot2.index.round(2))
    fig.colorbar(im2, ax=axes[i,1])

plt.tight_layout()
plt.savefig(out, dpi=150)
print(f'Wrote {out}')
