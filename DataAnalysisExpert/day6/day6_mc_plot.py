import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('day6_mc_results.csv')

# Humanity occurrence
plt.figure(figsize=(6,4))
plt.hist(df['humanity'].astype(int), bins=[-0.5,0.5,1.5], rwidth=0.6)
plt.xticks([0,1])
plt.title('Day-6 Humanity-image occurrence')
plt.savefig('day6_humanity_hist.png')
print('Wrote day6_humanity_hist.png')

# Dominion proxy distribution (log)
plt.figure(figsize=(6,4))
dom = df['dominion'].replace(0,1)
plt.hist(np.log10(dom), bins=60)
plt.xlabel('log10(dominion proxy)')
plt.title('Day-6 Dominion proxy (log10)')
plt.savefig('day6_dominion_hist.png')
print('Wrote day6_dominion_hist.png')
