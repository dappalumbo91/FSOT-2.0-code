import sys
import pandas as pd
import matplotlib.pyplot as plt

# Usage: python day6_hominid_plot.py <day6_mc_hominid.csv> <out_png>
if len(sys.argv) < 3:
    print('Usage: day6_hominid_plot.py <day6_mc_hominid.csv> <out_png>')
    sys.exit(1)

csv = sys.argv[1]
out = sys.argv[2]

df = pd.read_csv(csv)
# plot distributions for divergence, sapiens, behavioral modernity
fig, axes = plt.subplots(1,3, figsize=(15,4))

axes[0].hist(df['divergence_Ma'].dropna(), bins=40, color='C0')
axes[0].set_title('Hominid divergence (Ma)')
axes[0].set_xlabel('Ma')

axes[1].hist(df['sapiens_Ma'].dropna(), bins=40, color='C1')
axes[1].set_title('Sapiens emergence (Ma)')
axes[1].set_xlabel('Ma')

axes[2].hist(df['behavioral_Ma'].dropna(), bins=40, color='C2')
axes[2].set_title('Behavioral modernity (Ma)')
axes[2].set_xlabel('Ma')

plt.tight_layout()
plt.savefig(out, dpi=150)
print(f'Wrote {out}')
