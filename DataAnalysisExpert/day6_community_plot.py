import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('day6_mc_results.csv')

plt.figure(figsize=(6,4))
plt.hist(df['curiosity'].dropna(), bins=30, color='#6a91bf')
plt.xlabel('Curiosity proxy')
plt.ylabel('Count')
plt.title('Day-6 Curiosity distribution')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('day6_curiosity_hist.png')

plt.figure(figsize=(6,4))
plt.hist(df['manipulation'].dropna(), bins=40, color='#bf6a6a')
plt.xlabel('Manipulation (tool dominion)')
plt.ylabel('Count')
plt.title('Tool/Manipulation distribution')
plt.yscale('log')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('day6_manipulation_hist.png')

print('Wrote day6_curiosity_hist.png and day6_manipulation_hist.png')