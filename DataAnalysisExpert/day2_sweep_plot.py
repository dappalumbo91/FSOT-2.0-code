import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv = 'day2/data/day2_sweep_results.csv'
df = pd.read_csv(csv)

# Pivot for heatmaps: rows=delta_psi_base, cols=valve_scale
for metric in ['sep_rate', 'punc_rate']:
    fig, axes = plt.subplots(1, 3, figsize=(15,4), constrained_layout=True)
    thresholds = sorted(df['puncture_thresh'].unique())
    for ax, pt in zip(axes, thresholds):
        sub = df[df['puncture_thresh']==pt]
        pivot = sub.pivot(index='delta_psi_base', columns='valve_scale', values=metric)
        im = ax.imshow(pivot.values, origin='lower', cmap='viridis', vmin=0, vmax=100)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(int(c)) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(c) for c in pivot.index])
        ax.set_title(f"{metric} (puncture_thresh={pt})")
        ax.set_xlabel('valve_scale')
        ax.set_ylabel('delta_psi_base')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out = f"day2/images/day2_sweep_{metric}_heatmaps.png"
    fig.suptitle(f"Day-2 sweep — {metric}")
    fig.savefig(out)
    print('Wrote', out)

# Comparative line plot: punc_rate vs valve_scale for each delta_psi
plt.figure(figsize=(6,4))
for dp in sorted(df['delta_psi_base'].unique()):
    sub = df[df['delta_psi_base']==dp]
    means = sub.groupby('valve_scale')['punc_rate'].mean()
    plt.plot(means.index, means.values, marker='o', label=f"Δψ={dp}")
plt.xlabel('valve_scale')
plt.ylabel('punc_rate (%)')
plt.title('Puncture rate vs valve_scale (averaged over puncture_thresh)')
plt.legend()
plt.grid(True)
plt.savefig('day2_sweep_punc_vs_valvescale.png')
print('Wrote day2_sweep_punc_vs_valvescale.png')
