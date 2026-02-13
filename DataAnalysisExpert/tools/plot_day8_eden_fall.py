import sys
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print('Usage: plot_day8_eden_fall.py <csv> <out_png>')
    sys.exit(1)

csv = sys.argv[1]
out = sys.argv[2]

df = pd.read_csv(csv)
resets = df[df['reset_flag']==True]

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(resets['strife_length'], bins=30, color='C2', edgecolor='k')
plt.title('Strife episode lengths')
plt.xlabel('cycles')
plt.ylabel('count')

plt.subplot(1,2,2)
plt.hist(resets['fidelity_delta'], bins=30, color='C4', edgecolor='k')
plt.title('Fidelity delta during rebounds')
plt.xlabel('fidelity_delta (fraction)')
plt.ylabel('count')

plt.tight_layout()
plt.savefig(out, dpi=150)
print(f'Wrote {out}')
