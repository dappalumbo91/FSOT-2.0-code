import sys
import pandas as pd
import matplotlib.pyplot as plt

def plot_habitability(csv_path, out_png):
    df = pd.read_csv(csv_path)
    if 'habitability_score' not in df.columns:
        raise RuntimeError('CSV missing habitability_score')
    plt.figure(figsize=(6,4))
    plt.hist(df['habitability_score'].dropna(), bins=30, color='#2b7bba')
    plt.xlabel('Habitability score')
    plt.ylabel('Count')
    plt.title('Exoplanet habitability (FSOT proxy)')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_png)

def plot_eccentricity(csv_path, out_png):
    df = pd.read_csv(csv_path)
    if 'pl_orbeccen' not in df.columns and 'pl_orbeccen' not in df.columns:
        raise RuntimeError('CSV missing pl_orbeccen')
    vals = pd.to_numeric(df['pl_orbeccen'], errors='coerce').dropna()
    plt.figure(figsize=(6,4))
    plt.hist(vals, bins=40, color='#c95f5f')
    plt.xlabel('Eccentricity')
    plt.ylabel('Count')
    plt.title('Eccentricity distribution')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_png)

def plot_class_fraction(csv_path, out_png):
    df = pd.read_csv(csv_path)
    if 'class' not in df.columns:
        raise RuntimeError('CSV missing class column')
    counts = df['class'].value_counts()
    plt.figure(figsize=(6,4))
    counts.plot(kind='bar', color=['#2b7bba','#7fbf7f','#c95f5f','#b58fe6','#f0a43c'])
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Planet class fractions')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_png)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: planetary_db_plot.py <input.csv> <habitability.png> <eccentricity.png> [class_png]')
    else:
        inp = sys.argv[1]
        plot_habitability(inp, sys.argv[2])
        plot_eccentricity(inp, sys.argv[3])
        if len(sys.argv) > 4:
            plot_class_fraction(inp, sys.argv[4])
