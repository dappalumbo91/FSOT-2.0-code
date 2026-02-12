import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('day5_mc_results.csv')

# Swarming rate by habitability
plt.figure(figsize=(6,4))
plt.hist(df['swarming'].astype(int), bins=[-0.5,0.5,1.5], rwidth=0.6)
plt.xticks([0,1])
plt.xlabel('Swarming (0=no,1=yes)')
plt.ylabel('Count')
plt.title('Day-5 Swarming Outcomes')
plt.savefig('day5_swarming_hist.png')
print('Wrote day5_swarming_hist.png')

# Diversity histogram (log scale)
plt.figure(figsize=(6,4))
div = df['diversity'].replace(0,1)
plt.hist(np.log10(div), bins=60)
plt.xlabel('log10 Diversity proxy')
plt.ylabel('Count')
plt.title('Day-5 Diversity (log10)')
plt.savefig('day5_diversity_hist.png')
print('Wrote day5_diversity_hist.png')