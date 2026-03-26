import os
import re
import pandas as pd
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

def process_file(file_path):
    df = pd.read_pickle(file_path)
    #TODO: add here a way to read also the original df without having removed the 2 kev interactions
    #then check the accuracy also if we keep into account the 2 kev interactions that were lost (those events are not correctly groupable)

    # Group by grouped_ID
    grouped = df.groupby("grouped_ID")

    # Count correct groupings
    def is_group_correct(group, event_counts):
        unique_events = group["event_ID"].unique()
        if len(unique_events) != 1:
            return False
        event_id = unique_events[0]
        return len(group) == event_counts[event_id]

    event_counts = df.groupby("event_ID").size()
    #total_groups = len(grouped)
    total_correct = 0
    #total_primary = 0
    correct_primary = 0

    for _, group in grouped:
        correct = is_group_correct(group, event_counts)
        if correct:
            total_correct += 1
        if "photon_primary" in group.columns and group["photon_primary"].any():
            #total_primary += 1
            if correct:
                correct_primary += 1
    n_events = df['event_ID'].nunique()
    total_primary = df.groupby('event_ID')['photon_primary'].max().sum()
    accuracy = total_correct / n_events if n_events > 0 else 0
    primary_accuracy = correct_primary / total_primary if total_primary > 0 else 0
    #TODO: add here that it saves also accuracy and primary accuracy for the 2 kev case
    return accuracy, primary_accuracy


def extract_params(file_name):
    # Example: "..._no_noise_pdCollection time uncertainty (ns)_3_2_60_grouped.pkl"
    match = re.search(r"_no_noise_pd(.*?)_(\d+)_(\d+)_(\d+)_grouped", file_name)
    if not match:
        return None
    method = match.group(1)
    small_coeff = int(match.group(2))
    big_coeff = int(match.group(3))
    perc = int(match.group(4))
    return method, small_coeff, big_coeff, perc


def analyze_folder(folder):
    results = []
    output_csv = folder + ".csv"
    for file_name in os.listdir(folder):
        if file_name.endswith(".pkl"):
            params = extract_params(file_name)
            if not params:
                continue
            method, small_coeff, big_coeff, perc = params
            file_path = os.path.join(folder, file_name)
            acc, prim_acc = process_file(file_path)
            results.append({
                "method": method,
                "small_coeff": small_coeff,
                "big_coeff": big_coeff,
                "perc": perc,
                "accuracy": acc,
                "primary_accuracy": prim_acc
                # TODO: add here that it saves also accuracy and primary accuracy for the 2 kev case
            })

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    return df_results

def plot_heatmaps(df, method, perc):
    subset = df[(df["method"] == method) & (df["perc"] == perc)]

    pivot_acc = subset.pivot(index="small_coeff", columns="big_coeff", values="accuracy")
    pivot_prim = subset.pivot(index="small_coeff", columns="big_coeff", values="primary_accuracy")

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    sns.heatmap(pivot_acc, annot=True, fmt=".2f", cmap="viridis")
    plt.title(f"Accuracy ({method}, perc={perc})")

    plt.subplot(1,2,2)
    sns.heatmap(pivot_prim, annot=True, fmt=".2f", cmap="viridis")
    plt.title(f"Primary Accuracy ({method}, perc={perc})")

    # TODO: add here that it saves also accuracy and primary accuracy for the 2 kev caseadd here plots for this case

    plt.tight_layout()
    plt.show()


#analyze_folder("results_water_radii")
plot_heatmaps(pd.read_csv("results_water_radii.csv"), method="Collection time uncertainty (ns)", perc=90)
plot_heatmaps(pd.read_csv("results_water_radii.csv"), method="Jitter (ns) (newest)", perc=90) #Jitter (ns) (IB=200nA, PRE-HOLD)
plot_heatmaps(pd.read_csv("results_water_radii.csv"), method="Jitter (ns) (IB=200nA, PRE-HOLD)", perc=90)
plot_heatmaps(pd.read_csv("results_water_radii.csv"), method="Jitter (ns) (IB=400nA, PRE-HOLD)", perc=75)