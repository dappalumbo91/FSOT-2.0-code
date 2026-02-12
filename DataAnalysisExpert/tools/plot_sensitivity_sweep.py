import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

base = Path(__file__).resolve().parent.parent
fn = base / "sensitivity" / "sensitivity_sweep_summary.csv"
df = pd.read_csv(fn)

# plot grouped by multiplier (there are two rows per multiplier in the sweep)
groups = df.groupby('multiplier')
means = groups[['day4_habitable_frac','day6_human_rate','mean_dominion']].mean()
stds = groups[['day4_habitable_frac','day6_human_rate','mean_dominion']].std()

fig, ax1 = plt.subplots(figsize=(8,4))
ax1.errorbar(means.index, means['day4_habitable_frac'], yerr=stds['day4_habitable_frac'], marker='o', label='Day-4 habitable frac (%)')
ax1.set_ylabel('Day-4 habitable frac (%)')
ax1.set_ylim(0, 105)
ax1.set_xlabel('Multiplier (m)')
ax1.grid(alpha=0.2)

ax2 = ax1.twinx()
ax2.plot(means.index, means['day6_human_rate'], 's--', color='C1', label='Day-6 human rate (%)')
ax2.set_ylabel('Day-6 human rate (%)')
ax2.set_ylim(0, 105)

# dominion on log scale, separate small inset
ax3 = fig.add_axes([0.62, 0.55, 0.25, 0.33])
ax3.plot(means.index, means['mean_dominion'], 'k-o')
ax3.set_yscale('log')
ax3.set_title('Mean dominion (log)')
ax3.set_xlabel('m')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left')

plt.tight_layout()
out = base / 'sensitivity' / 'sensitivity_sweep_summary.png'
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150)
print(f'Wrote plot: {out}')
