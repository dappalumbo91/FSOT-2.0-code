import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_path = 'DataAnalysisExpert/day2/data/day2_sweep_results.csv'
df = pd.read_csv(csv_path)

# Heatmap: puncture probability vs Δψ and valve_scale (use first puncture_thresh)
pt = df['puncture_thresh'].unique()[0]
df_pt = df[df['puncture_thresh'] == pt]
unique_dp = np.sort(df_pt['delta_psi_base'].unique())
unique_vs = np.sort(df_pt['valve_scale'].unique())
mat = np.full((len(unique_dp), len(unique_vs)), np.nan)
for i, dp in enumerate(unique_dp):
    for j, vs in enumerate(unique_vs):
        row = df_pt[(df_pt['delta_psi_base'] == dp) & (df_pt['valve_scale'] == vs)]
        if not row.empty:
            mat[i, j] = row['punc_rate'].values[0]

plt.figure(figsize=(6,4))
plt.imshow(mat, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=100)
plt.colorbar(label='Puncture %')
plt.xticks(ticks=np.arange(len(unique_vs)), labels=unique_vs)
plt.yticks(ticks=np.arange(len(unique_dp)), labels=unique_dp)
plt.xlabel('Valve scale')
plt.ylabel('Δψ (yin bias)')
plt.title(f'Puncture probability (puncture_thresh={pt})')
plt.tight_layout()
plt.savefig('DataAnalysisExpert/day2/images/puncture_heatmap.png', dpi=150)
plt.close()

# Histogram: calibrated FRB energies (erg)
plt.figure(figsize=(6,4))
vals = df['mean_frb_erg'].dropna()
plt.hist(vals, bins=20, color='C1')
plt.xlabel('Mean FRB energy (erg)')
plt.ylabel('Count')
plt.title('Calibrated FRB energy (mean per combo)')
plt.tight_layout()
plt.savefig('DataAnalysisExpert/day2/images/frb_energy_hist.png', dpi=150)
plt.close()

# Scatter: Outgas mass vs e-folds
plt.figure(figsize=(6,4))
plt.scatter(df['mean_outgas_Msun'], df['mean_e_folds'], c='C2')
plt.xlabel('Mean outgas mass (M_sun proxy)')
plt.ylabel('Calibrated e-folds (proxy)')
plt.title('Outgas mass vs expansion proxy')
plt.tight_layout()
plt.savefig('DataAnalysisExpert/day2/images/quasar_scatter.png', dpi=150)
plt.close()

print('Saved: DataAnalysisExpert/day2/images/puncture_heatmap.png, DataAnalysisExpert/day2/images/frb_energy_hist.png, DataAnalysisExpert/day2/images/quasar_scatter.png')
