import os
import re
import pandas as pd
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

def process_file(file_path,file_original_path):
    df = pd.read_pickle(file_path)
    df_original = pd.read_pickle(file_original_path)
    #then check the accuracy also if we keep into account the 2 kev interactions that were lost (those events are not correctly groupable)

    # Group by grouped_ID
    grouped = df.groupby("grouped_ID")
    event_id_original = df_original.groupby("event_ID")
    # Count correct groupings
    def is_group_correct(group, event_counts):
        unique_events = group["event_ID"].unique()
        if len(unique_events) != 1:
            return False
        event_id = unique_events[0]
        return len(group) == event_counts[event_id]

    def is_group_correct_original(group, event_id_original):
        unique_events = group["event_ID"].unique()
        if len(unique_events) != 1:
            return False
        event_id = unique_events[0]
        event_id_original_group = event_id_original.get_group(event_id)
        return len(group) == len(event_id_original_group)

    event_counts = df.groupby("event_ID").size()
    #total_groups = len(grouped)
    total_correct = 0
    #total_primary = 0
    correct_primary = 0
    total_correct_2kev = 0
    correct_primary_2kev = 0
    for _, group in grouped:
        correct = is_group_correct(group, event_counts)
        if correct:
            total_correct += 1
        if "photon_primary" in group.columns and group["photon_primary"].any():
            #total_primary += 1
            if correct:
                correct_primary += 1
        correct_2kev = is_group_correct_original(group, event_id_original)
        if correct_2kev:
            total_correct_2kev += 1
        if "photon_primary" in group.columns and group["photon_primary"].any():
            if correct_2kev:
                correct_primary_2kev += 1




    n_events = df['event_ID'].nunique()
    total_primary = df.groupby('event_ID')['photon_primary'].max().sum()
    accuracy = total_correct / n_events if n_events > 0 else 0
    primary_accuracy = correct_primary / total_primary if total_primary > 0 else 0

    #TODO: the total primary grouped and the calculations about the primaries do not make much sense as of now need to fix
    total_primary_grouped = df.groupby('grouped_ID')['photon_primary'].max().sum()
    total_primary_incorrect = total_primary_grouped - correct_primary
    false_positive_primary = total_primary_incorrect / total_primary if total_primary > 0 else 0
    total_incorrect = df['grouped_ID'].nunique() - total_correct
    false_positive = total_incorrect / n_events if n_events > 0 else 0


    max_grouped_id = df['event_ID'].max()

    df_filtered = df_original[df_original["event_ID"] < max_grouped_id]
    n_events_2kev = df_filtered['event_ID'].nunique()
    total_primary_2kev = df_filtered.groupby('event_ID')['photon_primary'].max().sum()
    accuracy_2kev = total_correct_2kev / n_events_2kev if n_events_2kev > 0 else 0
    primary_accuracy_2kev = correct_primary_2kev / total_primary_2kev if total_primary_2kev > 0 else 0

    total_primary_incorrect_2kev = total_primary_grouped - correct_primary_2kev
    false_positive_2kev_primary = total_primary_incorrect_2kev / total_primary_2kev if total_primary_2kev > 0 else 0
    total_incorrect_2kev = df['grouped_ID'].nunique() - total_correct_2kev
    false_positive_2kev = total_incorrect_2kev / n_events_2kev if n_events_2kev > 0 else 0

    return accuracy, primary_accuracy, accuracy_2kev, primary_accuracy_2kev, false_positive, false_positive_primary, false_positive_2kev, false_positive_2kev_primary

def process_file_2_more_int(file_path,file_original_path):
    df = pd.read_pickle(file_path)
    df_original = pd.read_pickle(file_original_path)
    #then check the accuracy also if we keep into account the 2 kev interactions that were lost (those events are not correctly groupable)

    # Group by grouped_ID
    grouped = df.groupby("grouped_ID")
    event_id_original = df_original.groupby("event_ID")
    # Count correct groupings
    def is_group_correct(group, event_counts):
        unique_events = group["event_ID"].unique()
        if len(unique_events) != 1 or len(group)==1:
            return False
        event_id = unique_events[0]
        return len(group) == event_counts[event_id]

    def is_group_correct_original(group, event_id_original):
        unique_events = group["event_ID"].unique()
        if len(unique_events) != 1 or len(group)==1:
            return False
        event_id = unique_events[0]
        event_id_original_group = event_id_original.get_group(event_id)
        return len(group) == len(event_id_original_group)

    event_counts = df.groupby("event_ID").size()
    #total_groups = len(grouped)
    total_correct = 0
    #total_primary = 0
    correct_primary = 0
    total_correct_2kev = 0
    correct_primary_2kev = 0
    for _, group in grouped:
        correct = is_group_correct(group, event_counts)
        if correct:
            total_correct += 1
        if "photon_primary" in group.columns and group["photon_primary"].any():
            #total_primary += 1
            if correct:
                correct_primary += 1
        correct_2kev = is_group_correct_original(group, event_id_original)
        if correct_2kev:
            total_correct_2kev += 1
        if "photon_primary" in group.columns and group["photon_primary"].any():
            if correct_2kev:
                correct_primary_2kev += 1

    valid_ids = df.groupby("event_ID").filter(lambda g: len(g) >= 2)

    n_events = valid_ids["event_ID"].nunique()
    total_primary = valid_ids.groupby("event_ID")["photon_primary"].max().sum()

    accuracy = total_correct / n_events if n_events > 0 else 0
    primary_accuracy = correct_primary / total_primary if total_primary > 0 else 0

    max_grouped_id = df['event_ID'].max()

    df_filtered = df_original[df_original["event_ID"] < max_grouped_id]
    valid_ids_2kev = df_filtered.groupby("event_ID").filter(lambda g: len(g) >= 2)
    n_events_2kev = valid_ids_2kev['event_ID'].nunique()
    total_primary_2kev = valid_ids_2kev.groupby('event_ID')['photon_primary'].max().sum()
    accuracy_2kev = total_correct_2kev / n_events_2kev if n_events_2kev > 0 else 0
    primary_accuracy_2kev = correct_primary_2kev / total_primary_2kev if total_primary_2kev > 0 else 0


    return accuracy, primary_accuracy, accuracy_2kev, primary_accuracy_2kev

def extract_params(file_name):
    # Example: "..._no_noise_pdCollection time uncertainty (ns)_3_2_60_grouped.pkl"
    match = re.search(r"_no_noise_pd(.*?)_(\d+)_(\d+)_(\d+)_(\d+)_realiz_grouped", file_name)
    if not match:
        return None
    method = match.group(1)
    small_coeff = int(match.group(2))
    big_coeff = int(match.group(3))
    perc = int(match.group(4))
    realiz = int(match.group(5))
    return method, small_coeff, big_coeff, perc, realiz


def analyze_folder(folder,file_original_path):
    results = []
    output_csv = folder + ".csv"
    for file_name in os.listdir(folder):
        print('done file: ',file_name)
        if file_name.endswith(".pkl"):
            params = extract_params(file_name)
            if not params:
                continue
            method, small_coeff, big_coeff, perc, realiz = params
            file_path = os.path.join(folder, file_name)
            acc, prim_acc, acc_2kev, prim_acc_2kev, false_positive, false_positive_primary, false_positive_2kev, false_positive_2kev_primary = process_file(file_path, file_original_path)
            results.append({
                "method": method,
                "small_coeff": small_coeff,
                "big_coeff": big_coeff,
                "perc": perc,
                "realiz": realiz,
                "accuracy": acc,
                "primary_accuracy": prim_acc,
                "accuracy_2kev": acc_2kev,
                "primary_accuracy_2kev": prim_acc_2kev,
                "false_positive": false_positive,
                "false_positive_primary": false_positive_primary,
                "false_positive_2kev": false_positive_2kev,
                "false_positive_2kev_primary": false_positive_2kev_primary
            })

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    return df_results


def analyze_folder_2_more_int(folder,file_original_path):
    results = []
    output_csv = folder + "2_more_int.csv"
    for file_name in os.listdir(folder):
        print('done file: ',file_name)
        if file_name.endswith(".pkl"):
            params = extract_params(file_name)
            if not params:
                continue
            method, small_coeff, big_coeff, perc, realiz = params
            file_path = os.path.join(folder, file_name)
            acc, prim_acc, acc_2kev, prim_acc_2kev = process_file_2_more_int(file_path, file_original_path)
            results.append({
                "method": method,
                "small_coeff": small_coeff,
                "big_coeff": big_coeff,
                "perc": perc,
                "realiz": realiz,
                "accuracy": acc,
                "primary_accuracy": prim_acc,
                "accuracy_2kev": acc_2kev,
                "primary_accuracy_2kev": prim_acc_2kev
            })

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    return df_results

def plot_heatmaps(df, method, perc):
    subset = df[(df["method"] == method) & (df["perc"] == perc)]
    subset_mean = subset.groupby(["small_coeff", "big_coeff"], as_index=False).mean(numeric_only=True)

    pivot_acc = subset_mean.pivot(index="small_coeff", columns="big_coeff", values="accuracy")
    pivot_prim = subset_mean.pivot(index="small_coeff", columns="big_coeff", values="primary_accuracy")
    pivot_acc_2kev = subset_mean.pivot(index="small_coeff", columns="big_coeff", values="accuracy_2kev")
    pivot_prim_2kev = subset_mean.pivot(index="small_coeff", columns="big_coeff", values="primary_accuracy_2kev")

    plt.figure(figsize=(12,5))
    plt.subplot(2,2,1)
    sns.heatmap(pivot_acc, annot=True, fmt=".2f", cmap="viridis")
    plt.title(f"Accuracy ({method}, perc={perc})")

    plt.subplot(2,2,2)
    sns.heatmap(pivot_prim, annot=True, fmt=".2f", cmap="viridis")
    plt.title(f"Primary Accuracy ({method}, perc={perc})")

    plt.subplot(2,2,3)
    sns.heatmap(pivot_acc_2kev, annot=True, fmt=".2f", cmap="viridis")
    plt.title(f"Accuracy 1_5kev ({method}, perc={perc})")

    plt.subplot(2,2,4)
    sns.heatmap(pivot_prim_2kev, annot=True, fmt=".2f", cmap="viridis")
    plt.title(f"Primary Accuracy 1_5ev ({method}, perc={perc})")


    plt.tight_layout()
    plt.show()

def plot_heatmaps_false_positive(df, method, perc):
    subset = df[(df["method"] == method) & (df["perc"] == perc)]
    subset_mean = subset.groupby(["small_coeff", "big_coeff"], as_index=False).mean(numeric_only=True)

    pivot_acc = subset_mean.pivot(index="small_coeff", columns="big_coeff", values="false_positive")
    pivot_prim = subset_mean.pivot(index="small_coeff", columns="big_coeff", values="false_positive_primary")
    pivot_acc_2kev = subset_mean.pivot(index="small_coeff", columns="big_coeff", values="false_positive_2kev")
    pivot_prim_2kev = subset_mean.pivot(index="small_coeff", columns="big_coeff", values="false_positive_2kev_primary")

    plt.figure(figsize=(12,5))
    plt.subplot(2,2,1)
    sns.heatmap(pivot_acc, annot=True, fmt=".2f", cmap="viridis")
    plt.title(f"False positive ({method}, perc={perc})")

    #plt.subplot(2,2,2)
    #sns.heatmap(pivot_prim, annot=True, fmt=".2f", cmap="viridis")
    #plt.title(f"False_positive_primary ({method}, perc={perc})")

    plt.subplot(2,2,3)
    sns.heatmap(pivot_acc_2kev, annot=True, fmt=".2f", cmap="viridis")
    plt.title(f"False_positive_2kev ({method}, perc={perc})")

    #plt.subplot(2,2,4)
    #sns.heatmap(pivot_prim_2kev, annot=True, fmt=".2f", cmap="viridis")
    #plt.title(f"False_positive_2kev_primary ({method}, perc={perc})")


    plt.tight_layout()
    plt.show()

def p_values_comparison(df,method2, perc,method1 = "Collection time uncertainty (ns)"):
    method1_subset = df[(df["method"] == method1) & (df["perc"] == perc) & (df["small_coeff"] == 4) & (df["big_coeff"] == 2)]
    method2_subset = df[(df["method"] == method2) & (df["perc"] == perc) & (df["small_coeff"] == 4) & (df["big_coeff"] == 2)]

    acc_method1 = method1_subset["accuracy"]
    acc_method2 = method2_subset["accuracy"]
    t_stat_acc, p_value_acc = ttest_ind(acc_method1, acc_method2, alternative="two-sided")

    acc_primary_method1 = method1_subset["primary_accuracy"]
    acc_primary_method2 = method2_subset["primary_accuracy"]
    t_stat_primary, p_value_primary = ttest_ind(acc_primary_method1, acc_primary_method2, alternative="two-sided")

    acc_2kev_method1 = method1_subset["accuracy_2kev"]
    acc_2kev_method2 = method2_subset["accuracy_2kev"]
    t_stat_acc_2kev, p_value_acc_2kev = ttest_ind(acc_2kev_method1, acc_2kev_method2, alternative="two-sided")

    acc_primary_2kev_method1 = method1_subset["primary_accuracy_2kev"]
    acc_primary_2kev_method2 = method2_subset["primary_accuracy_2kev"]
    t_stat_primary_2kev, p_value_primary_2kev = ttest_ind(acc_primary_2kev_method1, acc_primary_2kev_method2, alternative="two-sided")

    print(f"P-value for accuracy between {method1} and {method2} at perc={perc}: {p_value_acc}")
    print(f"P-value for primary accuracy between {method1} and {method2} at perc={perc}: {p_value_primary}")
    print(f"P-value for accuracy 2kev between {method1} and {method2} at perc={perc}: {p_value_acc_2kev}")
    print(f"P-value for primary accuracy 2kev between {method1} and {method2} at perc={perc}: {p_value_primary_2kev}")


    return

############################################################################
#analyze_folder("results_water_radii_realizations",'full_detector_bulk_140kev_5x20x20cm_20R_100H_water_phantom500_MBq_no_noise_pd.pkl')
plot_heatmaps(pd.read_csv("results_water_radii_realizations.csv"), method="Collection time uncertainty (ns)", perc=75)
plot_heatmaps(pd.read_csv("results_water_radii_realizations.csv"), method="Jitter (ns) (newest)", perc=75) #Jitter (ns) (IB=200nA, PRE-HOLD)
plot_heatmaps(pd.read_csv("results_water_radii_realizations.csv"), method="Jitter (ns) (IB=200nA, PRE-HOLD)", perc=75)
plot_heatmaps(pd.read_csv("results_water_radii_realizations.csv"), method="Jitter (ns) (IB=400nA, PRE-HOLD)", perc=75)
plot_heatmaps_false_positive(pd.read_csv("results_water_radii_realizations.csv"), method="Collection time uncertainty (ns)", perc=75)
plot_heatmaps_false_positive(pd.read_csv("results_water_radii_realizations.csv"), method="Jitter (ns) (newest)", perc=75) #Jitter (ns) (IB=200nA, PRE-HOLD)
plot_heatmaps_false_positive(pd.read_csv("results_water_radii_realizations.csv"), method="Jitter (ns) (IB=200nA, PRE-HOLD)", perc=75)
plot_heatmaps_false_positive(pd.read_csv("results_water_radii_realizations.csv"), method="Jitter (ns) (IB=400nA, PRE-HOLD)", perc=75)


#p_values_comparison(pd.read_csv("results_water_radii_realizations.csv"), method2="Jitter (ns) (newest)", perc=75)
#p_values_comparison(pd.read_csv("results_water_radii_realizations.csv"), method2="Jitter (ns) (IB=200nA, PRE-HOLD)", perc=75)
#p_values_comparison(pd.read_csv("results_water_radii_realizations.csv"), method2="Jitter (ns) (IB=400nA, PRE-HOLD)", perc=75)


#analyze_folder_2_more_int("results_water_radii_realizations",'full_detector_bulk_140kev_5x20x20cm_20R_100H_water_phantom500_MBq_no_noise_pd.pkl')
#plot_heatmaps(pd.read_csv("results_water_radii_realizations2_more_int.csv"), method="Collection time uncertainty (ns)", perc=75)
#plot_heatmaps(pd.read_csv("results_water_radii_realizations2_more_int.csv"), method="Jitter (ns) (newest)", perc=75) #Jitter (ns) (IB=200nA, PRE-HOLD)
#plot_heatmaps(pd.read_csv("results_water_radii_realizations2_more_int.csv"), method="Jitter (ns) (IB=200nA, PRE-HOLD)", perc=75)
#plot_heatmaps(pd.read_csv("results_water_radii_realizations2_more_int.csv"), method="Jitter (ns) (IB=400nA, PRE-HOLD)", perc=75)

#p_values_comparison(pd.read_csv("results_water_radii_realizations.csv"), method2="Jitter (ns) (newest)", perc=75)
#p_values_comparison(pd.read_csv("results_water_radii_realizations.csv"), method2="Jitter (ns) (IB=200nA, PRE-HOLD)", perc=75)
#p_values_comparison(pd.read_csv("results_water_radii_realizations.csv"), method2="Jitter (ns) (IB=400nA, PRE-HOLD)", perc=75)



#analyze_folder("results_no_water_radii_distr",'full_detector_bulk_140kev_5x20x20cm_20R_100H_water_phantom500_MBq_no_noise_pd.pkl')
#plot_heatmaps(pd.read_csv("results_no_water_radii_distr.csv"), method="Collection time uncertainty (ns)", perc=75)
#plot_heatmaps(pd.read_csv("results_no_water_radii_distr.csv"), method="Jitter (ns) (newest)", perc=75) #Jitter (ns) (IB=200nA, PRE-HOLD)
#plot_heatmaps(pd.read_csv("results_no_water_radii_distr.csv"), method="Jitter (ns) (IB=200nA, PRE-HOLD)", perc=75)
#plot_heatmaps(pd.read_csv("results_no_water_radii_distr.csv"), method="Jitter (ns) (IB=400nA, PRE-HOLD)", perc=75)

#analyze_folder("results_water_radii_realizations_1_5_kev_th",'full_detector_bulk_140kev_5x20x20cm_20R_100H_water_phantom500_MBq_no_noise_pd.pkl')
#plot_heatmaps(pd.read_csv("results_water_radii_realizations_1_5_kev_th.csv"), method="Collection time uncertainty (ns)", perc=75)
#plot_heatmaps(pd.read_csv("results_water_radii_realizations_1_5_kev_th.csv"), method="Jitter (ns) (newest)", perc=75) #Jitter (ns) (IB=200nA, PRE-HOLD)
#plot_heatmaps(pd.read_csv("results_water_radii_realizations_1_5_kev_th.csv"), method="Jitter (ns) (IB=200nA, PRE-HOLD)", perc=75)
#plot_heatmaps(pd.read_csv("results_water_radii_realizations_1_5_kev_th.csv"), method="Jitter (ns) (IB=400nA, PRE-HOLD)", perc=75)
############################################################################