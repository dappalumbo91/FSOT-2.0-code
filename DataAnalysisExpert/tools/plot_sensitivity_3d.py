import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Usage: python plot_sensitivity_3d.py <sweep_csv> <out_png>
if len(sys.argv) < 3:
    print("Usage: plot_sensitivity_3d.py <sweep_csv> <out_png>")
    sys.exit(1)

csv = sys.argv[1]
out = sys.argv[2]

df = pd.read_csv(csv)
# prefer day5_swarm or day4_hab_rate
if 'day5_swarm' in df.columns:
    zval = df['day5_swarm']
elif 'day4_hab_rate' in df.columns:
    zval = df['day4_hab_rate']
else:
    zval = np.zeros(len(df))

x = df.get('tectonic_mult', df.get('tectonic_rate', df.get('tectonic', np.nan)))
y = df.get('ocean_frac', df.get('ocean_fraction', np.nan))
z = df.get('weather_idx', df.get('weather_index', np.nan))

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z, c=zval, cmap='viridis', depthshade=True, s=60, alpha=0.9)
ax.set_xlabel('tectonic multiplier / rate')
ax.set_ylabel('ocean fraction')
ax.set_zlabel('weather index')
plt.title('Sensitivity: tectonic vs ocean vs weather (color = outcome)')
cb = plt.colorbar(sc, pad=0.1)
cb.set_label('Outcome (day5_swarm or day4_hab_rate)')
plt.tight_layout()
plt.savefig(out, dpi=150)
print(f"Wrote {out}")
