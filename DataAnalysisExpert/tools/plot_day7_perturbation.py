import sys
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print('Usage: plot_day7_perturbation.py <csv> <out_png>')
    sys.exit(1)

csv = sys.argv[1]
out = sys.argv[2]

df = pd.read_csv(csv)
resets = df[df['reset_flag'] == True]['reset_strength']

plt.figure(figsize=(6,4))
plt.hist(resets, bins=20, color='C1', edgecolor='k')
plt.title('Day-7 Perturbation Reset Strengths')
plt.xlabel('reset_strength')
plt.ylabel('count')
plt.tight_layout()
plt.savefig(out, dpi=150)
print(f'Wrote {out}')
