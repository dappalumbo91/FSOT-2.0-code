import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Usage: export_day8_timeline.py <narrative_csv> <fidelity_csv> <out_annotated_csv> <out_highlights_csv> <out_png>
if len(sys.argv) < 6:
    print('Usage: export_day8_timeline.py <narrative_csv> <fidelity_csv> <out_annotated_csv> <out_highlights_csv> <out_png>')
    sys.exit(1)

narr_csv = sys.argv[1]
fid_csv = sys.argv[2]
out_annot = sys.argv[3]
out_hl = sys.argv[4]
out_png = sys.argv[5]

narr = pd.read_csv(narr_csv)
fid = pd.read_csv(fid_csv)

# join fidelity into narrative rows (ensure same epoch ordering)
df = narr.merge(fid, left_on='epoch', right_on='epoch', how='left')

# cumulative fidelity already in fidelity column; compute pct change over prior epoch
df['fidelity_pct_change'] = df['fidelity'].pct_change().fillna(0.0) * 100.0

# dominion proxy: use a scaled proxy from fidelity (simple heuristic)
df['dominion_proxy'] = (df['fidelity'] / df['fidelity'].median()).clip(lower=0.01)

# annotate covenant renewal events: reset_flag & rebound_success & positive fidelity_delta
conditions = [ (df['reset_flag']==True) & (df['rebound_success']==True) & (df['fidelity_after']>df['fidelity_before']),
               (df['reset_flag']==True) & (df['rebound_success']==True) & (df['fidelity_after']<=df['fidelity_before']),
               (df['reset_flag']==True) & (df['rebound_success']==False) ]
choices = ['covenant_renewal', 'reset_rebound_neutral_or_loss', 'reset_failed_rebound']
df['event_type'] = np.select(conditions, choices, default='none')

# flag high-impact events: large positive fidelity change > mean + 2*std among positive rebounds
pos_changes = df.loc[df['event_type']=='covenant_renewal', 'fidelity_after'] - df.loc[df['event_type']=='covenant_renewal', 'fidelity_before']
if len(pos_changes) > 0:
    threshold = pos_changes.mean() + 2*pos_changes.std()
else:
    threshold = 0.0

# mark highlighted covenant bursts
df['highlight'] = ((df['event_type']=='covenant_renewal') & ((df['fidelity_after'] - df['fidelity_before']) > threshold))

# create highlights summary (top covenant renewals by fidelity delta)
hl = df[df['event_type']=='covenant_renewal'].copy()
hl['fidelity_delta'] = hl['fidelity_after'] - hl['fidelity_before']
hl_sorted = hl.sort_values('fidelity_delta', ascending=False)
highlights = hl_sorted[['epoch','strife_length','fidelity_before','fidelity_after','fidelity_delta','event_type','highlight']].head(50)

# export annotated timeline and highlights
df.to_csv(out_annot, index=False)
highlights.to_csv(out_hl, index=False)

# quick plot: fidelity time series with highlights
plt.figure(figsize=(12,4))
plt.plot(df['epoch'], df['fidelity'], lw=0.7, label='fidelity')
# mark covenant renewals
pts = df[df['highlight']]
plt.scatter(pts['epoch'], pts['fidelity'], color='C3', s=30, label='highlight covenant bursts')
# mark resets
resets = df[df['reset_flag']==True]
plt.scatter(resets['epoch'], resets['fidelity'], color='C1', s=6, alpha=0.6, label='resets')
plt.xlabel('epoch')
plt.ylabel('fidelity (relative)')
plt.title('Day-8 Narrative — annotated timeline highlights')
plt.legend()
plt.tight_layout()
plt.savefig(out_png, dpi=150)
print(f'Wrote annotated timeline: {out_annot}, highlights: {out_hl}, plot: {out_png}')
