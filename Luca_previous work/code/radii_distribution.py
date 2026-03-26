import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
def compute_radii_by_event_size(df, event_id_col, pos_cols):
    """
    For each unique event ID, compute the interaction radius (max distance from centroid).
    Store radii in a dict keyed by the number of interactions per event.
    """
    radii_by_size = defaultdict(list)
    grouped = df.groupby(event_id_col)
    for event_id, group in grouped:
        coords = np.asarray(group[pos_cols].values)
        centroid = np.mean(coords, axis=0)
        distances = np.linalg.norm(np.asarray(coords - centroid,dtype=float), axis=1)
        radius = np.max(distances)
        n_interactions = len(group)
        radii_by_size[n_interactions].append(radius)
    return dict(radii_by_size)

df = pd.read_pickle('full_detector_bulk_140kev_5x20x20cm500_MBq_no_noise_pd_radii.pkl')
df = df[df['E_dep'] >= 0.0001].reset_index(drop=True)
radii_distribution = compute_radii_by_event_size(df, 'event_ID', ['init_x', 'init_y', 'init_z'])

plt.figure(figsize=(10,6))

# Convert dict to list-of-lists
data = [radii_distribution[k] for k in sorted(radii_distribution.keys())]

plt.boxplot(data, positions=sorted(radii_distribution.keys()))
plt.xlabel("Number of interactions per event")
plt.ylabel("Radius")
plt.title("Radius Distributions by Event Size (Boxplot)")
plt.grid(True, alpha=0.3)
plt.show()

sizes = []
means = []
stds = []

for n_interactions, radii in radii_distribution.items():
    sizes.append(n_interactions)
    means.append(np.mean(radii))
    stds.append(np.std(radii))

plt.errorbar(sizes, means, yerr=stds, fmt='o-', capsize=3)
plt.xlabel("Number of interactions per event")
plt.ylabel("Radius")
plt.title("Mean Radius ± Std by Event Size")
plt.grid(True, alpha=0.3)
plt.show()



with open('radii_distribution_no_water.pkl', 'wb') as f:
    pickle.dump(radii_distribution, f)