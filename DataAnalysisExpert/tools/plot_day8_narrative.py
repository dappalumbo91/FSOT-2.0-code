import sys
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print('Usage: plot_day8_narrative.py <csv> <out_png>')
    sys.exit(1)

csv = sys.argv[1]
out = sys.argv[2]

df = pd.read_csv(csv)
plt.figure(figsize=(10,4))
plt.plot(df['epoch'], df['fidelity'], lw=0.6)
plt.xlabel('epoch')
plt.ylabel('fidelity (relative)')
plt.title('Day-8 Narrative — fidelity over epochs')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out, dpi=150)
print(f'Wrote {out}')
