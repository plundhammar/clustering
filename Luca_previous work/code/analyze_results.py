import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def process_file(file_path):

    data = pd.read_pickle(os.path.join("results",file_path))  # Assuming CSV format

    # 1. Count the number of event_IDs of different lengths
    event_counts = data.groupby('event_ID').size().value_counts().sort_index()

    # 2. Count the number of group_IDs of different lengths
    group_counts = data.groupby('grouped_ID').size().value_counts().sort_index()

    # 3. Check how many group_IDs are correct for each chain length
    def is_group_correct(group, event_counts):
        unique_events = group['event_ID'].unique()
        if len(unique_events) != 1:
            return False  # The group contains multiple event_IDs
        event_id = unique_events[0]
        return len(group) == event_counts[event_id]  # The group must contain all interactions of that event_ID

    grouped = data.groupby('grouped_ID')
    correct_counts = {length: 0 for length in group_counts.index}

    event_interaction_counts = data.groupby('event_ID').size()

    for _, group in grouped:
        length = len(group)
        if is_group_correct(group, event_interaction_counts):
            correct_counts[length] += 1

    correct_counts_series = pd.Series(correct_counts).sort_index()

    return event_counts, group_counts, correct_counts_series


# Process 4 files
file_paths = ["full_detector_bulk_140kev_5x20x20cm_20R_100H_water_phantom500_MBq_no_noise_pdJitter (ns) (newest)_3_2_75_grouped.pkl",
              "full_detector_bulk_140kev_5x20x20cm_20R_100H_water_phantom500_MBq_no_noise_pdCollection time uncertainty (ns)_3_2_60_grouped.pkl",
              "full_detector_bulk_140kev_5x20x20cm_20R_100H_water_phantom500_MBq_no_noise_pdCollection time uncertainty (ns)_3_2_90_grouped.pkl",
              "full_detector_bulk_140kev_5x20x20cm_20R_100H_water_phantom500_MBq_no_noise_pdCollection time uncertainty (ns)_3_3_75_grouped.pkl",
              "full_detector_bulk_140kev_5x20x20cm_20R_100H_water_phantom500_MBq_no_noise_pdCollection time uncertainty (ns)_2_3_75_grouped.pkl",
              "full_detector_bulk_140kev_5x20x20cm_20R_100H_water_phantom500_MBq_no_noise_pdCollection time uncertainty (ns)_3_2_75_grouped.pkl"]
event_counts_list, group_counts_list, correct_counts_list = [], [], []

for file in file_paths:
    event_counts, group_counts, correct_counts_series = process_file(file)
    event_counts_list.append(event_counts)
    correct_counts_list.append(correct_counts_series)
    group_counts_list.append(group_counts)

# Plot event counts from first file with group counts for all 4 files
plt.figure(figsize=(10, 6))
plt.bar(event_counts_list[0].index[:7], event_counts_list[0].values[:7], alpha=0.6, label='Event Counts (truth)')

for i, group_counts in enumerate(group_counts_list):
    plt.plot(group_counts.index[:7], group_counts.values[:7], linestyle='-', marker='o',
             label=f'{file_paths[i]}')

plt.xlabel("Interaction Length")
plt.ylabel("Counts")
plt.legend()
plt.title("Event Counts and Group Counts")
plt.show()

# Plot correct_counts_series as percentage of event_counts
plt.figure(figsize=(10, 6))
for i, group_counts in enumerate(group_counts_list):
    percentage = (correct_counts_list[i] / event_counts_list[i]) * 100
    plt.plot(percentage.index[:7], percentage.values[:7], linestyle='-', marker='o', label=f'{file_paths[i]}')

plt.xlabel("Interaction Length")
plt.ylabel("Correct Grouping Percentage (%)")
plt.legend()
plt.title("Correct Grouping Percentage per Interaction Length")
plt.show()
