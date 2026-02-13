import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 3:
    print('Usage: plot_day7_sweep.py <csv> <out_png>')
    sys.exit(1)

csv = sys.argv[1]
out = sys.argv[2]

df = pd.read_csv(csv)
# pivot for heatmaps
pivot_rest = df.pivot_table(index='curiosity_scale', columns='community_strength', values='rest_rate')
pivot_cycle = df.pivot_table(index='curiosity_scale', columns='community_strength', values='cycle_rate')

fig, axes = plt.subplots(1,2, figsize=(12,5))
im0 = axes[0].imshow(pivot_rest.values, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=100)
axes[0].set_title('Day-7 Rest-stable rate (%)')
axes[0].set_xlabel('community_strength')
axes[0].set_ylabel('curiosity_scale')
axes[0].set_xticks(np.arange(pivot_rest.shape[1])); axes[0].set_xticklabels(pivot_rest.columns.round(2))
axes[0].set_yticks(np.arange(pivot_rest.shape[0])); axes[0].set_yticklabels(pivot_rest.index.round(2))
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(pivot_cycle.values, origin='lower', aspect='auto', cmap='plasma', vmin=0, vmax=100)
axes[1].set_title('Day-7 Cycle-stable rate (%)')
axes[1].set_xlabel('community_strength')
axes[1].set_ylabel('curiosity_scale')
axes[1].set_xticks(np.arange(pivot_cycle.shape[1])); axes[1].set_xticklabels(pivot_cycle.columns.round(2))
axes[1].set_yticks(np.arange(pivot_cycle.shape[0])); axes[1].set_yticklabels(pivot_cycle.index.round(2))
fig.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.savefig(out, dpi=150)
print(f'Wrote {out}')
