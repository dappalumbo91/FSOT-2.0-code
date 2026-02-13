import sys
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print('Usage: plot_day7.py <day7_csv> <out_png>')
    sys.exit(1)

csv = sys.argv[1]
out = sys.argv[2]

df = pd.read_csv(csv)
fig, ax = plt.subplots(1,2, figsize=(10,4))
ax[0].hist(df['S'].dropna(), bins=40, color='C0')
ax[0].set_title('Day‑7 S distribution (rest noise)')
ax[0].set_xlabel('S')
ax[1].hist(df['observer_fidelity'].dropna(), bins=40, color='C1')
ax[1].set_title('Observer fidelity proxy')
ax[1].set_xlabel('fidelity')
plt.tight_layout()
plt.savefig(out, dpi=150)
print(f'Wrote {out}')
